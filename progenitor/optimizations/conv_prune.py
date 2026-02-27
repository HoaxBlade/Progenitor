"""
Structured filter pruning for Conv layers in CNNs (e.g. ResNet, VGG).

For ResNet-style bottleneck blocks:
  Conv1x1 (reduce) -> Relu -> Conv3x3 (process) -> Relu -> Conv1x1 (expand) [+ shortcut -> Add -> Relu]

We prune the bottleneck (middle) Conv3x3 — reduce its output AND input channels.
This requires adjusting the output of the preceding Conv1x1 and the input of the following Conv1x1.

The residual branch dimensions are left untouched so the Add node still works.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, numpy_helper


def _build_graph_maps(model: ModelProto):
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    producer = {}
    consumers = {}
    for n in model.graph.node:
        for out in n.output:
            producer[out] = n
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    return init_map, producer, consumers


def _follow_relu(tensor, consumers):
    """If tensor feeds a single Relu, return that Relu's output. Otherwise return tensor."""
    ds = consumers.get(tensor, [])
    if len(ds) == 1 and ds[0].op_type == "Relu":
        return ds[0].output[0]
    return None


def _find_bottleneck_triples(model, init_map, consumers):
    """
    Find Conv1x1 -> Relu -> Conv3x3 -> Relu -> Conv1x1 triples in the graph.
    Returns list of (conv_reduce, conv_mid, conv_expand) node triples.
    """
    triples = []
    used = set()

    for node in model.graph.node:
        if node.op_type != "Conv" or node.input[1] not in init_map:
            continue
        w = init_map[node.input[1]]
        if w.ndim != 4:
            continue
        kH, kW = w.shape[2], w.shape[3]
        if not (kH == 3 and kW == 3):
            continue

        mid_node = node
        mid_w = w
        mid_out = mid_node.output[0]

        # mid_node should have a single consumer chain: Relu -> Conv1x1
        relu_out = _follow_relu(mid_out, consumers)
        if relu_out is None:
            continue
        expand_consumers = consumers.get(relu_out, [])
        if len(expand_consumers) != 1 or expand_consumers[0].op_type != "Conv":
            continue
        expand_node = expand_consumers[0]
        if expand_node.input[1] not in init_map:
            continue
        expand_w = init_map[expand_node.input[1]]
        if expand_w.ndim != 4 or expand_w.shape[2] != 1 or expand_w.shape[3] != 1:
            continue
        if expand_node.input[0] != relu_out:
            continue

        # mid_node's activation input should come from Conv1x1 -> Relu
        mid_act_input = mid_node.input[0]
        # This should be the output of a Relu whose input is Conv1x1 output
        mid_act_producer_list = [n for n in model.graph.node if n.output[0] == mid_act_input]
        if not mid_act_producer_list:
            continue
        mid_act_producer = mid_act_producer_list[0]
        if mid_act_producer.op_type != "Relu":
            continue
        relu_input = mid_act_producer.input[0]
        reduce_producer_list = [n for n in model.graph.node if n.output[0] == relu_input]
        if not reduce_producer_list:
            continue
        reduce_node = reduce_producer_list[0]
        if reduce_node.op_type != "Conv" or reduce_node.input[1] not in init_map:
            continue
        reduce_w = init_map[reduce_node.input[1]]
        if reduce_w.ndim != 4 or reduce_w.shape[2] != 1 or reduce_w.shape[3] != 1:
            continue

        key = (id(reduce_node), id(mid_node), id(expand_node))
        if key in used:
            continue
        used.add(key)

        triples.append((reduce_node, mid_node, expand_node))

    return triples


def apply_conv_structured_pruning(model: ModelProto, ratio: float) -> None:
    """
    Prune bottleneck Conv3x3 channels in ResNet-style blocks.
    Removes `ratio` fraction of channels from the middle 3x3 conv in each bottleneck.
    
    This correspondingly adjusts the output channels of the preceding 1x1 conv
    and the input channels of the following 1x1 conv.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")

    init_map, producer, consumers = _build_graph_maps(model)
    triples = _find_bottleneck_triples(model, init_map, consumers)

    if not triples:
        return

    updates = {}

    for reduce_node, mid_node, expand_node in triples:
        reduce_w_name = reduce_node.input[1]
        mid_w_name = mid_node.input[1]
        expand_w_name = expand_node.input[1]

        reduce_w = init_map[reduce_w_name]  # (bottleneck_ch, in_ch, 1, 1)
        mid_w = init_map[mid_w_name]        # (bottleneck_ch, bottleneck_ch, 3, 3)
        expand_w = init_map[expand_w_name]  # (out_ch, bottleneck_ch, 1, 1)

        bottleneck_ch = mid_w.shape[0]
        n_keep = max(1, int(bottleneck_ch * (1 - ratio)))
        if n_keep >= bottleneck_ch:
            continue

        # Score mid_w filters by L2 norm across spatial+input dims
        scores = np.linalg.norm(mid_w.reshape(bottleneck_ch, -1), axis=1)
        keep_idx = np.argsort(scores)[-n_keep:]
        keep_idx.sort()

        # Mid conv: prune both output and input channels (it's square: bottleneck_ch x bottleneck_ch)
        new_mid_w = mid_w[keep_idx][:, keep_idx]
        updates[mid_w_name] = new_mid_w

        # Mid bias
        if len(mid_node.input) > 2 and mid_node.input[2] in init_map:
            mid_b_name = mid_node.input[2]
            updates[mid_b_name] = init_map[mid_b_name][keep_idx]

        # Reduce conv: prune output channels (these feed the mid conv's input)
        reduce_w_cur = updates.get(reduce_w_name, reduce_w)
        updates[reduce_w_name] = reduce_w_cur[keep_idx]

        # Reduce bias
        if len(reduce_node.input) > 2 and reduce_node.input[2] in init_map:
            reduce_b_name = reduce_node.input[2]
            reduce_b = updates.get(reduce_b_name, init_map[reduce_b_name])
            updates[reduce_b_name] = reduce_b[keep_idx]

        # Expand conv: prune input channels (these come from mid conv's output)
        expand_w_cur = updates.get(expand_w_name, expand_w)
        updates[expand_w_name] = expand_w_cur[:, keep_idx]

    if not updates:
        return

    new_inits = []
    for init in model.graph.initializer:
        if init.name in updates:
            arr = updates[init.name].astype(np.float32)
            new_inits.append(numpy_helper.from_array(arr, init.name))
        else:
            new_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    del model.graph.value_info[:]

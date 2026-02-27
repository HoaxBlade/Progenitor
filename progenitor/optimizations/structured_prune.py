"""
Structured pruning: remove entire neurons from weight matrices, physically shrinking the graph.

Unlike unstructured pruning (which zeros weights but keeps the same shapes), this
changes the dimensions of MatMul/Gemm ops so the model is genuinely smaller and
runs faster on any runtime (no sparse backend needed).

For an MLP with layers W1(in, H), W2(H, H), W3(H, out):
  - Removing neuron j from hidden layer means deleting column j from W1 (or the
    preceding weight), row j from W2, and element j from bias1.
  - The resulting graph has smaller dimensions and fewer FLOPs.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper


def _find_linear_chain(model: ModelProto) -> list[dict]:
    """
    Parse an MLP-like ONNX graph into a list of layers.
    Each layer: {w_name, bias_name (or None), relu, node, w_arr, bias_arr}.
    Assumes topological order and MatMul/Gemm + optional Add + optional Relu pattern.
    """
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    ready = set(inp.name for inp in model.graph.input) | set(init_map.keys())
    order = []
    remaining = list(model.graph.node)
    while remaining:
        for i, n in enumerate(remaining):
            if all(inp in ready for inp in n.input):
                order.append(n)
                ready |= set(n.output)
                remaining.pop(i)
                break
        else:
            raise ValueError("Graph has cycles or missing inputs")

    layers = []
    i = 0
    while i < len(order):
        node = order[i]
        if node.op_type == "Gemm":
            w_name = node.input[1] if node.input[1] in init_map else node.input[0]
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] in init_map else None
            has_relu = (i + 1 < len(order) and order[i + 1].op_type == "Relu")
            layers.append(dict(
                w_name=w_name, bias_name=bias_name, relu=has_relu,
                w_arr=init_map[w_name].astype(np.float32),
                bias_arr=init_map[bias_name].astype(np.float32) if bias_name else None,
            ))
            if has_relu:
                i += 1
            i += 1
        elif node.op_type == "MatMul":
            w_name = None
            for idx in (0, 1):
                if node.input[idx] in init_map and init_map[node.input[idx]].ndim == 2:
                    w_name = node.input[idx]
                    break
            if w_name is None:
                i += 1
                continue
            bias_name = None
            has_relu = False
            if i + 1 < len(order) and order[i + 1].op_type == "Add":
                add_node = order[i + 1]
                for inp in add_node.input:
                    if inp in init_map and init_map[inp].ndim == 1:
                        bias_name = inp
                        break
                i += 1
                if i + 1 < len(order) and order[i + 1].op_type == "Relu":
                    has_relu = True
                    i += 1
            elif i + 1 < len(order) and order[i + 1].op_type == "Relu":
                has_relu = True
                i += 1
            layers.append(dict(
                w_name=w_name, bias_name=bias_name, relu=has_relu,
                w_arr=init_map[w_name].astype(np.float32),
                bias_arr=init_map[bias_name].astype(np.float32) if bias_name else None,
            ))
            i += 1
        else:
            i += 1
    return layers


def apply_structured_pruning(model: ModelProto, ratio: float) -> None:
    """
    Remove `ratio` fraction of hidden neurons from each internal layer.
    Neurons are scored by L2 norm of their outgoing weights.
    Modifies model in-place: updates initializer shapes/values and graph I/O shapes.

    ratio: 0.5 = remove 50% of hidden neurons -> ~2-4x fewer FLOPs.
    Only internal layers are pruned (first layer output and last layer output kept intact).
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")

    layers = _find_linear_chain(model)
    if len(layers) < 2:
        return

    init_map = {i.name: i for i in model.graph.initializer}

    # For each pair of consecutive layers, prune the connecting dimension.
    # Layer i has W_i (in_i, out_i). The "hidden" dim is out_i = in_{i+1}.
    # We score neurons by L2 norm of column j in W_i (the outgoing weights).
    for li in range(len(layers) - 1):
        cur = layers[li]
        nxt = layers[li + 1]
        W_cur = cur["w_arr"]   # (in_cur, hidden)
        W_nxt = nxt["w_arr"]   # (hidden, out_nxt)
        hidden = W_cur.shape[1]
        n_keep = max(1, int(hidden * (1 - ratio)))

        # Score each neuron by L2 norm of its outgoing column in W_cur
        scores = np.linalg.norm(W_cur, axis=0)  # (hidden,)
        keep_idx = np.argsort(scores)[-n_keep:]  # top n_keep by score
        keep_idx.sort()

        # Prune W_cur columns
        W_cur_new = W_cur[:, keep_idx]
        cur["w_arr"] = W_cur_new

        # Prune bias of cur layer
        if cur["bias_arr"] is not None:
            cur["bias_arr"] = cur["bias_arr"][keep_idx]

        # Prune W_nxt rows
        W_nxt_new = W_nxt[keep_idx, :]
        nxt["w_arr"] = W_nxt_new

    # Write pruned weights back to model initializers
    new_inits = []
    updated = {}
    for layer in layers:
        updated[layer["w_name"]] = layer["w_arr"]
        if layer["bias_name"] is not None and layer["bias_arr"] is not None:
            updated[layer["bias_name"]] = layer["bias_arr"]

    for init in model.graph.initializer:
        if init.name in updated:
            arr = updated[init.name]
            new_init = numpy_helper.from_array(arr.astype(np.float32), init.name)
            new_inits.append(new_init)
        else:
            new_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Clear stale value_info shapes — dimensions changed, so ORT's shape
    # inference needs to recompute them from scratch.
    del model.graph.value_info[:]

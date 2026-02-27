"""
Conv Channel Pruning — physically remove output channels from Conv kernels.

For each Conv layer, scores output channels by L2 norm of their kernel weights,
removes the weakest channels, and cascades the dimension change to:
  - The Conv's bias (if present)
  - The downstream Conv's input channels

Safety: Conv layers whose output feeds into an Add node (residual connections)
are SKIPPED to preserve dimension compatibility on skip connections.
This means only the interior bottleneck Convs (e.g. the 3x3 conv in a ResNet
bottleneck) are pruned, which is the standard approach used by real pruning
frameworks.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, numpy_helper


def _build_maps(model: ModelProto):
    """Build producer and consumer maps for the graph."""
    producers: dict[str, any] = {}
    consumers: dict[str, list] = {}
    inits: dict[str, np.ndarray] = {}

    for init in model.graph.initializer:
        inits[init.name] = numpy_helper.to_array(init)

    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            if inp not in consumers:
                consumers[inp] = []
            consumers[inp].append(node)

    return producers, consumers, inits


def _set_initializer(model: ModelProto, name: str, arr: np.ndarray):
    """Replace an initializer's data in the model."""
    for init in model.graph.initializer:
        if init.name == name:
            new_tensor = numpy_helper.from_array(arr, name=name)
            init.CopyFrom(new_tensor)
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def _feeds_into_add(node, consumers) -> bool:
    """Check if a node's output feeds into an Add node (directly or via Relu/BN)."""
    visited = set()

    def _check(tensor_name):
        if tensor_name in visited:
            return False
        visited.add(tensor_name)
        for consumer in consumers.get(tensor_name, []):
            if consumer.op_type == "Add":
                return True
            if consumer.op_type in ("Relu", "BatchNormalization"):
                for out in consumer.output:
                    if _check(out):
                        return True
        return False

    for out in node.output:
        if _check(out):
            return True
    return False


def _find_downstream_convs(node, consumers) -> list:
    """Walk forward from a node's output to find directly downstream Conv nodes."""
    results = []
    visited = set()

    def _walk(tensor_name):
        if tensor_name in visited:
            return
        visited.add(tensor_name)
        for consumer in consumers.get(tensor_name, []):
            if consumer.op_type == "Conv":
                if consumer.input[0] == tensor_name:
                    results.append(consumer)
            elif consumer.op_type in ("Relu", "Add", "BatchNormalization",
                                       "MaxPool", "AveragePool",
                                       "GlobalAveragePool"):
                for out in consumer.output:
                    _walk(out)

    for out in node.output:
        _walk(out)
    return results


def apply_conv_channel_pruning(model: ModelProto, ratio: float = 0.5) -> int:
    """
    Prune output channels from Conv layers.

    Skip-connection-safe: Conv layers whose output feeds into an Add node
    are NOT pruned, preserving residual connection dimension compatibility.

    Args:
        model: ONNX model to modify in-place.
        ratio: fraction of channels to REMOVE (0.5 = remove 50%).

    Returns:
        Number of Conv layers pruned.
    """
    if ratio <= 0 or ratio >= 1:
        return 0

    producers, consumers, inits = _build_maps(model)

    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]

    # Track which Conv layers were pruned and their keep masks
    pruned_masks: dict[str, np.ndarray] = {}  # conv_name -> keep_indices
    pruned_count = 0

    for conv in conv_nodes:
        w_name = conv.input[1]
        if w_name not in inits:
            continue

        W = inits[w_name]  # (OC, IC, KH, KW)
        if W.ndim != 4:
            continue

        # SAFETY: skip Conv layers that feed into Add (residual connections)
        if _feeds_into_add(conv, consumers):
            continue

        OC = W.shape[0]
        keep_count = max(1, int(OC * (1.0 - ratio)))
        if keep_count >= OC:
            continue

        # Score channels by L2 norm of entire kernel slice
        scores = np.linalg.norm(W.reshape(OC, -1), axis=1)
        keep_indices = np.sort(np.argsort(scores)[-keep_count:])

        # Prune the Conv weight: W[OC, IC, KH, KW] -> W[keep, IC, KH, KW]
        W_pruned = W[keep_indices]
        _set_initializer(model, w_name, W_pruned.astype(np.float32))
        inits[w_name] = W_pruned

        # Prune bias if present
        if len(conv.input) > 2 and conv.input[2] in inits:
            b_name = conv.input[2]
            bias = inits[b_name]
            bias_pruned = bias[keep_indices]
            _set_initializer(model, b_name, bias_pruned.astype(np.float32))
            inits[b_name] = bias_pruned

        pruned_masks[conv.name] = keep_indices

        # Cascade: prune input channels of downstream Conv layers
        downstream_convs = _find_downstream_convs(conv, consumers)
        for ds_conv in downstream_convs:
            ds_w_name = ds_conv.input[1]
            if ds_w_name not in inits:
                continue
            ds_W = inits[ds_w_name]
            if ds_W.ndim != 4:
                continue

            # Check group attribute
            group = 1
            for attr in ds_conv.attribute:
                if attr.name == "group":
                    group = attr.i
                    break

            if group > 1:
                # Depthwise conv (group == OC, IC_per_g == 1)
                if group == ds_W.shape[0] and ds_W.shape[1] == 1:
                    ds_W_pruned = ds_W[keep_indices]
                    for attr in ds_conv.attribute:
                        if attr.name == "group":
                            attr.i = len(keep_indices)
                            break
                else:
                    continue
            else:
                # Regular conv: prune input channels (axis 1)
                if ds_W.shape[1] != W.shape[0]:
                    continue  # Dimension mismatch
                ds_W_pruned = ds_W[:, keep_indices, :, :]

            _set_initializer(model, ds_w_name, ds_W_pruned.astype(np.float32))
            inits[ds_w_name] = ds_W_pruned

        pruned_count += 1

    return pruned_count

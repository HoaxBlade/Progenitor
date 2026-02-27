"""Magnitude-based weight pruning for sparse inference (5–15× target on same CPU)."""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, numpy_helper

# Ops that have weight initializers we can prune (input indices that are weights)
_WEIGHT_INPUT_INDICES: dict[str, list[int]] = {
    "Conv": [1],           # W
    "MatMul": [0, 1],      # A, B
    "Gemm": [0, 1],        # A, B
}


def _weight_initializer_names(model: ModelProto) -> set[str]:
    """Names of initializers used as weights by Conv/MatMul/Gemm."""
    names: set[str] = set()
    initializer_names = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        indices = _WEIGHT_INPUT_INDICES.get(node.op_type)
        if indices is None:
            continue
        for i in indices:
            if i < len(node.input) and node.input[i] in initializer_names:
                names.add(node.input[i])
    return names


def _prune_array(arr: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero out smallest-magnitude elements so that sparsity fraction of elements are zero."""
    if sparsity <= 0 or arr.size == 0:
        return arr
    if sparsity >= 1:
        return np.zeros_like(arr)
    flat = np.abs(arr.ravel())
    n_keep = max(1, int((1 - sparsity) * flat.size))
    # threshold = (n_keep)th largest magnitude = (size - n_keep)th smallest
    k = flat.size - n_keep
    if k <= 0:
        return arr.copy()
    threshold = np.partition(flat, k)[k]
    out = arr.copy()
    out[np.abs(out) < threshold] = 0
    return out


def apply_pruning(model: ModelProto, sparsity: float) -> None:
    """
    Prune weight initializers in-place: zero out smallest-magnitude weights to achieve
    target sparsity (e.g. 0.9 = 90% zeros). Only Conv/MatMul/Gemm weight tensors are pruned.
    Same graph, same ops; use with sparse inference for 5–15× speedup.
    """
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be in [0, 1]")
    to_prune = _weight_initializer_names(model)
    if not to_prune:
        return

    new_initializers = []
    for init in model.graph.initializer:
        if init.name not in to_prune:
            new_initializers.append(init)
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            new_initializers.append(init)
            continue
        if arr.size < 2:
            new_initializers.append(init)
            continue
        # Only prune float types
        if arr.dtype not in (np.float32, np.float64):
            new_initializers.append(init)
            continue
        pruned = _prune_array(arr, sparsity)
        new_init = numpy_helper.from_array(pruned.astype(arr.dtype), init.name)
        new_initializers.append(new_init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

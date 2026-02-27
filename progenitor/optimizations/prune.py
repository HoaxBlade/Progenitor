"""Magnitude-based and importance-based weight pruning for sparse inference (5–15× target on same CPU)."""

from __future__ import annotations

import copy
import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

# Ops that have weight initializers we can prune (input indices that are weights)
_WEIGHT_INPUT_INDICES: dict[str, list[int]] = {
    "Conv": [1],           # W
    "MatMul": [0, 1],      # A, B
    "Gemm": [0, 1],        # A, B
}


def _weight_to_data_tensor(model: ModelProto) -> dict[str, tuple[str, int]]:
    """Map weight initializer name -> (data_tensor_name, input_axis). input_axis is weight axis aligned with activation dim."""
    out: dict[str, tuple[str, int]] = {}
    initializer_names = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if len(node.input) < 2:
            continue
        for w_idx in (0, 1):
            if node.input[w_idx] not in initializer_names:
                continue
            data_tensor = node.input[1 - w_idx]
            if node.op_type == "MatMul":
                input_axis = 1 - w_idx
            else:
                trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
                input_axis = 1 if (w_idx == 0 or trans_b) else 0
            out[node.input[w_idx]] = (data_tensor, input_axis)
    return out


def compute_activation_importance(model: ModelProto, num_runs: int = 25) -> dict[str, np.ndarray]:
    """
    Run calibration to get per-dimension activation importance for tensors that feed MatMul/Gemm.
    Returns dict: tensor_name -> 1d array (norm per dimension). Used for importance-based pruning.
    """
    import onnxruntime as ort
    from onnx import shape_inference
    from progenitor.loader import save_onnx
    from progenitor.runner import create_random_feed

    weight_to_data = _weight_to_data_tensor(model)
    if not weight_to_data:
        return {}
    data_names = list(set(d[0] for d in weight_to_data.values()))
    init_names = {i.name for i in model.graph.initializer}
    graph_input_names = {inp.name for inp in model.graph.input}
    data_names = [n for n in data_names if n not in init_names]
    data_names_intermediate = [n for n in data_names if n not in graph_input_names]

    model_cal = copy.deepcopy(model)
    try:
        model_cal = shape_inference.infer_shapes(model_cal)
    except Exception:
        pass
    name_to_shape = {}
    for vi in model_cal.graph.value_info:
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            name_to_shape[vi.name] = dims
    for inp in model_cal.graph.input:
        if inp.type.tensor_type.shape.dim:
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                name_to_shape[inp.name] = dims

    for name in data_names_intermediate:
        if name not in name_to_shape or any(o.name == name for o in model_cal.graph.output):
            continue
        dims = name_to_shape[name]
        model_cal.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
        )

    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = Path(f.name)
    try:
        save_onnx(model_cal, tmp)
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        accum: dict[str, np.ndarray] = {}
        for _ in range(num_runs):
            feed = create_random_feed(sess)
            # Importance for graph inputs: use feed
            for name in data_names:
                if name in graph_input_names and name in feed:
                    arr = np.asarray(feed[name], dtype=np.float64)
                    if arr.ndim >= 2:
                        norms = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=0)
                    elif arr.ndim == 1:
                        norms = np.abs(arr)
                    else:
                        continue
                    if name not in accum:
                        accum[name] = np.zeros_like(norms)
                    accum[name] += norms
            if not data_names_intermediate:
                continue
            outs = sess.run(None, feed)
            out_names = [o.name for o in sess.get_outputs()]
            for name, arr in zip(out_names, outs):
                if name not in data_names_intermediate:
                    continue
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim >= 2:
                    norms = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=0)
                elif arr.ndim == 1:
                    norms = np.abs(arr)
                else:
                    continue
                if name not in accum:
                    accum[name] = np.zeros_like(norms)
                accum[name] += norms
        return {k: (v / num_runs).astype(np.float32) for k, v in accum.items()}
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _prune_array_by_importance(
    arr: np.ndarray, sparsity: float, importance_1d: np.ndarray, input_axis: int
) -> np.ndarray:
    """Zero weights with smallest |w| * importance[dim] to achieve sparsity. Keeps same sparsity as magnitude prune but better cosine."""
    if sparsity <= 0 or arr.size == 0:
        return arr
    if sparsity >= 1:
        return np.zeros_like(arr)
    imp = np.asarray(importance_1d, dtype=np.float64).ravel()
    if imp.size == 0:
        return arr.copy()
    score = np.abs(arr.astype(np.float64))
    if input_axis == 0:
        if score.shape[0] != imp.size:
            return arr.copy()
        score = score * imp[:, np.newaxis]
    else:
        if score.shape[1] != imp.size:
            return arr.copy()
        score = score * imp[np.newaxis, :]
    n_keep = max(1, int((1 - sparsity) * score.size))
    k = score.size - n_keep
    if k <= 0:
        return arr.copy()
    threshold = np.partition(score.ravel(), k)[k]
    out = arr.copy()
    out[score < threshold] = 0
    return out


def apply_importance_pruning(
    model: ModelProto,
    sparsity: float,
    importance: dict[str, np.ndarray],
) -> None:
    """
    Prune by importance (|w| * activation_importance) so same sparsity gives higher cosine.
    Modifies model in-place. Falls back to magnitude for weights without importance.
    """
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be in [0, 1]")
    to_prune = _weight_initializer_names(model)
    if not to_prune:
        return
    weight_to_data = _weight_to_data_tensor(model)
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
        if arr.size < 2 or arr.dtype not in (np.float32, np.float64):
            new_initializers.append(init)
            continue
        if init.name in weight_to_data:
            data_tensor, input_axis = weight_to_data[init.name]
            imp = importance.get(data_tensor)
            if imp is not None and np.issubdtype(np.asarray(imp).dtype, np.floating):
                imp = np.asarray(imp).ravel()
                if (input_axis == 0 and imp.size == arr.shape[0]) or (input_axis == 1 and imp.size == arr.shape[1]):
                    pruned = _prune_array_by_importance(arr, sparsity, imp, input_axis)
                    new_initializers.append(numpy_helper.from_array(pruned.astype(arr.dtype), init.name))
                    continue
        pruned = _prune_array(arr, sparsity)
        new_initializers.append(numpy_helper.from_array(pruned.astype(arr.dtype), init.name))

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)


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

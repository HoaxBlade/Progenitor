"""
Post-prune output calibration: fit scale and bias so pruned output best matches original.
Recovers cosine without fine-tuning by solving least squares on calibration runs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from progenitor.loader import save_onnx
from progenitor.runner import create_random_feed


def _collect_outputs(
    original_path: str | Path,
    pruned_path: str | Path,
    num_samples: int = 50,
    seed: int = 42,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Run both models on the same feeds; return (list of orig outputs per sample, list of pruned per sample)."""
    import onnxruntime as ort

    if seed is not None:
        np.random.seed(seed)
    sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    sess_pruned = ort.InferenceSession(str(pruned_path), providers=["CPUExecutionProvider"])

    orig_outputs: list[list[np.ndarray]] = []
    pruned_outputs: list[list[np.ndarray]] = []

    for _ in range(num_samples):
        feed = create_random_feed(sess_orig)
        out_orig = sess_orig.run(None, feed)
        out_pruned = sess_pruned.run(None, feed)
        orig_outputs.append([np.asarray(a).flatten().astype(np.float64) for a in out_orig])
        pruned_outputs.append([np.asarray(a).flatten().astype(np.float64) for a in out_pruned])

    return orig_outputs, pruned_outputs


def _fit_scale_bias(orig_list: list[np.ndarray], pruned_list: list[np.ndarray]) -> tuple[float, float]:
    """Least squares: scale * pruned + bias ≈ orig. Returns (scale, bias) scalar."""
    O = np.stack(orig_list, axis=0)
    P = np.stack(pruned_list, axis=0)
    P_flat = P.ravel()
    O_flat = O.ravel()
    A = np.stack([P_flat, np.ones_like(P_flat)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, O_flat, rcond=None)
    return float(x[0]), float(x[1])


def _fit_scale_bias_per_dim(orig_list: list[np.ndarray], pruned_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension least squares: scale[j]*pruned[:,j] + bias[j] ≈ orig[:,j]. Returns (scale_1d, bias_1d)."""
    O = np.stack(orig_list, axis=0)
    P = np.stack(pruned_list, axis=0)
    n, d = O.shape
    scale = np.ones(d, dtype=np.float64)
    bias = np.zeros(d, dtype=np.float64)
    for j in range(d):
        A = np.stack([P[:, j], np.ones(n)], axis=1)
        x, _, _, _ = np.linalg.lstsq(A, O[:, j], rcond=None)
        scale[j], bias[j] = float(x[0]), float(x[1])
    return scale.astype(np.float32), bias.astype(np.float32)


def apply_output_calibration(
    original_model_path: str | Path,
    pruned_model: ModelProto,
    *,
    num_samples: int = 50,
    seed: int = 42,
) -> None:
    """
    After pruning, run original vs pruned on the same inputs and fit scale/bias per output
    so that scale * pruned_output + bias ≈ original_output. Add Mul+Add nodes to the pruned
    model so inference produces the calibrated output. Modifies pruned_model in place.
    Improves cosine (output direction) without full fine-tuning.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_pruned = Path(f.name)
    try:
        save_onnx(pruned_model, tmp_pruned)
        orig_outputs, pruned_outputs = _collect_outputs(
            original_model_path, tmp_pruned, num_samples=num_samples, seed=seed
        )
    finally:
        try:
            tmp_pruned.unlink()
        except FileNotFoundError:
            pass

    n_outs = len(pruned_model.graph.output)
    if n_outs == 0:
        return

    # Transpose to list per output: [out0_samples, out1_samples, ...]
    orig_per_out = [[orig_outputs[s][i] for s in range(num_samples)] for i in range(n_outs)]
    pruned_per_out = [[pruned_outputs[s][i] for s in range(num_samples)] for i in range(n_outs)]

    # Per-dimension scale/bias so direction (cosine) can be corrected, not just norm/offset.
    scales_biases = [
        _fit_scale_bias_per_dim(orig_per_out[i], pruned_per_out[i]) for i in range(n_outs)
    ]

    # Append calibration nodes (Mul + Add) per output; replace only graph.output, do not remove existing nodes.
    graph = pruned_model.graph
    new_outputs = []
    name_used = {n.name for n in graph.node} | {i.name for i in graph.initializer}

    def unique_name(prefix: str) -> str:
        base = prefix
        c = 0
        while base in name_used:
            base = f"{prefix}_{c}"
            c += 1
        name_used.add(base)
        return base

    for i, out_info in enumerate(graph.output):
        old_out_name = out_info.name
        scale_arr, bias_arr = scales_biases[i]
        scale_name = unique_name("calib_scale")
        bias_name = unique_name("calib_bias")
        mul_out = unique_name("calib_mul_out")
        new_out_name = unique_name("calib_final_out")

        graph.initializer.append(
            numpy_helper.from_array(scale_arr, scale_name)
        )
        graph.initializer.append(
            numpy_helper.from_array(bias_arr, bias_name)
        )

        graph.node.append(
            helper.make_node("Mul", [scale_name, old_out_name], [mul_out], name=unique_name("CalibMul"))
        )
        graph.node.append(
            helper.make_node("Add", [mul_out, bias_name], [new_out_name], name=unique_name("CalibAdd"))
        )

        dims = out_info.type.tensor_type.shape.dim
        shape = []
        for d in dims:
            if getattr(d, "dim_value", 0) > 0:
                shape.append(d.dim_value)
            elif getattr(d, "dim_param", None):
                shape.append(d.dim_param)
            else:
                shape.append(1)
        new_outputs.append(
            helper.make_tensor_value_info(new_out_name, out_info.type.tensor_type.elem_type, shape)
        )

    del graph.output[:]
    graph.output.extend(new_outputs)

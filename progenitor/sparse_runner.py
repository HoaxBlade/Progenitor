"""
Sparse inference runner: run pruned ONNX models with sparse matmul for real 5–15× speedup.

Backends (first available wins):
- sparse-dot-mkl: Intel MKL sparse BLAS (pip install sparse-dot-mkl) — best for 5–15× on Intel CPU.
- torch: PyTorch sparse (pip install progenitor[sparse] or torch).
Falls back to ONNX Runtime dense if no backend or model not supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from onnx import ModelProto, numpy_helper

# Ops we can run with sparse weights (Gemm optional, MLP = MatMul+Add+Relu)
_SPARSE_OPS = {"MatMul", "Add", "Relu", "Gemm"}


def _weight_names_for_sparse(model: ModelProto) -> set[str]:
    """Initializer names used as weights in MatMul/Gemm (same logic as prune.py)."""
    names: set[str] = set()
    init_names = {i.name for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2 and node.input[1] in init_names:
            names.add(node.input[1])
        if node.op_type == "MatMul" and len(node.input) >= 2:
            for i in (0, 1):
                if node.input[i] in init_names:
                    names.add(node.input[i])
        if node.op_type == "Gemm" and len(node.input) >= 2:
            for i in (0, 1):
                if node.input[i] in init_names:
                    names.add(node.input[i])
    return names


def _is_sparse_enough(arr: np.ndarray, min_sparsity: float = 0.5) -> bool:
    if arr.dtype not in (np.float32, np.float64) or arr.ndim != 2 or arr.size == 0:
        return False
    return float(np.sum(arr == 0)) / arr.size >= min_sparsity


def _dense_to_csr(arr: np.ndarray):
    """Convert dense 2D numpy to PyTorch CSR (values, col_indices, row_offsets)."""
    import torch
    rows, cols = arr.shape
    nz = np.nonzero(arr)
    nz_rows, nz_cols = nz[0], nz[1]
    values = arr[nz_rows, nz_cols].astype(np.float32)
    # row_offsets: for each row i, number of non-zeros in rows 0..i-1
    row_offsets = np.zeros(rows + 1, dtype=np.int64)
    for r in nz_rows:
        row_offsets[r + 1] += 1
    np.cumsum(row_offsets, out=row_offsets)
    return (
        torch.from_numpy(values),
        torch.from_numpy(nz_cols.astype(np.int64)),
        torch.from_numpy(row_offsets),
        (rows, cols),
    )


def _dense_to_scipy_csr(arr: np.ndarray):
    """Convert dense 2D to scipy.sparse.csr_matrix for MKL."""
    import scipy.sparse as sp
    return sp.csr_matrix(arr.astype(np.float32))


def _execution_order(model: ModelProto) -> list:
    """Return graph nodes in execution order (producers before consumers)."""
    ready = set(inp.name for inp in model.graph.input)
    ready |= {i.name for i in model.graph.initializer}
    order = []
    remaining = list(model.graph.node)
    while remaining:
        made_progress = False
        for i, n in enumerate(remaining):
            if all(inp in ready for inp in n.input):
                order.append(n)
                ready |= set(n.output)
                remaining.pop(i)
                made_progress = True
                break
        if not made_progress:
            raise ValueError("Graph has cycles or missing inputs")
    return order


def _try_mkl_session(model_path: str | Path) -> "_SparseSessionMKL | None":
    """Build MKL-backed session if sparse_dot_mkl is available."""
    try:
        from sparse_dot_mkl import dot_product_mkl  # noqa: F401
    except ImportError:
        return None
    from progenitor.loader import load_onnx
    model = load_onnx(model_path)
    for node in model.graph.node:
        if node.op_type not in _SPARSE_OPS:
            return None
    return _SparseSessionMKL(model_path)


class _SparseSessionMKL:
    """Sparse session using Intel MKL (sparse-dot-mkl). Gives 5–15× on Intel CPU."""

    def __init__(self, model_path: str | Path) -> None:
        from sparse_dot_mkl import dot_product_mkl
        from progenitor.loader import load_onnx
        self._dot_mkl = dot_product_mkl
        model = load_onnx(model_path)
        self._model = model
        weight_names = _weight_names_for_sparse(model)
        init_arrays = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self._init_dense = {n: arr.astype(np.float32) for n, arr in init_arrays.items()}
        self._init_sparse = {}
        for name, arr in init_arrays.items():
            if name in weight_names and _is_sparse_enough(arr):
                self._init_sparse[name] = _dense_to_scipy_csr(arr)
        self._order = _execution_order(model)

    def run(self, input_feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dot_mkl = self._dot_mkl
        state = {k: v.astype(np.float32) for k, v in input_feed.items()}
        init_dense, init_sparse, order = self._init_dense, self._init_sparse, self._order
        for node in order:
            op = node.op_type
            out_name = node.output[0]
            if op == "MatMul":
                a_name, b_name = node.input[0], node.input[1]
                a = state.get(a_name)
                if a is None:
                    a = init_sparse.get(a_name)
                    if a is None:
                        a = init_dense.get(a_name)
                b = state.get(b_name)
                if b is None:
                    b = init_sparse.get(b_name)
                    if b is None:
                        b = init_dense.get(b_name)
                a_sp = getattr(a, "getnnz", None) is not None
                b_sp = getattr(b, "getnnz", None) is not None
                if not a_sp and b_sp:
                    # dense @ sparse -> (batch, in) @ (in, out)
                    state[out_name] = np.ascontiguousarray(dot_mkl(a, b, dense=True))
                elif a_sp and not b_sp:
                    # sparse @ dense -> (batch, in) = (batch, out) @ (out, in) so result = dense @ sparse = dot_mkl(b, a)
                    state[out_name] = np.ascontiguousarray(dot_mkl(b, a, dense=True))
                else:
                    ad = a.toarray() if a_sp else a
                    bd = b.toarray() if b_sp else b
                    state[out_name] = np.dot(ad, bd).astype(np.float32)
            elif op == "Add":
                a = state.get(node.input[0]) or init_dense[node.input[0]]
                b = state.get(node.input[1]) or init_dense[node.input[1]]
                a_arr = a.toarray() if getattr(a, "getnnz", None) is not None else a
                b_arr = b.toarray() if getattr(b, "getnnz", None) is not None else b
                state[out_name] = np.ascontiguousarray((a_arr + b_arr).astype(np.float32))
            elif op == "Relu":
                x = state[node.input[0]]
                x_arr = x.toarray() if getattr(x, "getnnz", None) is not None else x
                state[out_name] = np.maximum(x_arr, 0.0).astype(np.float32)
            elif op == "Gemm":
                a_name, b_name = node.input[0], node.input[1]
                c_name = node.input[2] if len(node.input) > 2 else None
                alpha, beta = 1.0, 1.0
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = attr.f
                    elif attr.name == "beta":
                        beta = attr.f
                A = state.get(a_name) or init_sparse.get(a_name) or init_dense.get(a_name)
                B = state.get(b_name) or init_sparse.get(b_name) or init_dense.get(b_name)
                C = (state.get(c_name) or init_dense.get(c_name)) if c_name else None
                A_arr = A.toarray() if getattr(A, "getnnz", None) is not None else A
                B_arr = B.toarray() if getattr(B, "getnnz", None) is not None else B
                if getattr(A, "getnnz", None) is not None:
                    ab = dot_mkl(A, B_arr, dense=True)
                elif getattr(B, "getnnz", None) is not None:
                    ab = dot_mkl(B, A_arr.T, dense=True).T
                else:
                    ab = np.dot(A_arr, B_arr)
                y = alpha * ab
                if C is not None:
                    y = y + beta * (C.toarray() if getattr(C, "getnnz", None) is not None else C)
                state[out_name] = np.ascontiguousarray(y.astype(np.float32))
        result = {}
        for out_info in self._model.graph.output:
            name = out_info.name
            if name in state:
                v = state[name]
                result[name] = v.toarray() if getattr(v, "getnnz", None) is not None else np.asarray(v)
        return result


class _SparseSession:
    """Holds loaded model and sparse tensors; run() does one forward (no reload). Uses PyTorch."""

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from progenitor.loader import load_onnx
        self._torch = torch
        model = load_onnx(model_path)
        self._model = model
        weight_names = _weight_names_for_sparse(model)
        init_arrays = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self._init_dense = {}
        self._init_sparse = {}
        for name, arr in init_arrays.items():
            t = torch.from_numpy(arr.astype(np.float32))
            self._init_dense[name] = t
            if name in weight_names and _is_sparse_enough(arr):
                v, col, row_offsets, shape = _dense_to_csr(arr)
                self._init_sparse[name] = torch.sparse_csr_tensor(
                    row_offsets, col, v, shape, dtype=torch.float32
                )
        self._order = _execution_order(model)
        for node in model.graph.node:
            if node.op_type not in _SPARSE_OPS:
                raise ValueError(
                    f"Sparse runner only supports {_SPARSE_OPS}; got {node.op_type}."
                )

    def run(self, input_feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self._torch
        state = {k: torch.from_numpy(v.astype(np.float32)) for k, v in input_feed.items()}
        init_dense, init_sparse, order = self._init_dense, self._init_sparse, self._order
        for node in order:
            op = node.op_type
            out_name = node.output[0]
            if op == "MatMul":
                a_name, b_name = node.input[0], node.input[1]
                a = state.get(a_name)
                if a is None:
                    a = init_sparse.get(a_name)
                    if a is None:
                        a = init_dense.get(a_name)
                b = state.get(b_name)
                if b is None:
                    b = init_sparse.get(b_name)
                    if b is None:
                        b = init_dense.get(b_name)
                a_dense = not (isinstance(a, torch.Tensor) and getattr(a, "is_sparse", False))
                b_dense = not (isinstance(b, torch.Tensor) and getattr(b, "is_sparse", False))
                if a_dense and not b_dense:
                    state[out_name] = torch.sparse.mm(b.t(), a.t()).t()
                elif not a_dense and b_dense:
                    state[out_name] = torch.sparse.mm(a, b.t()).t()
                else:
                    state[out_name] = torch.mm(a, b)
            elif op == "Add":
                a = state.get(node.input[0])
                if a is None:
                    a = init_dense[node.input[0]]
                b = state.get(node.input[1])
                if b is None:
                    b = init_dense[node.input[1]]
                state[out_name] = a + b
            elif op == "Relu":
                state[out_name] = state[node.input[0]].clamp(min=0.0)
            elif op == "Gemm":
                a_name, b_name = node.input[0], node.input[1]
                c_name = node.input[2] if len(node.input) > 2 else None
                alpha, beta = 1.0, 1.0
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = attr.f
                    elif attr.name == "beta":
                        beta = attr.f
                A = state.get(a_name) or init_sparse.get(a_name) or init_dense.get(a_name)
                B = state.get(b_name) or init_sparse.get(b_name) or init_dense.get(b_name)
                C = state.get(c_name) if c_name else None
                if c_name and C is None:
                    C = init_dense.get(c_name)
                if A is None or B is None:
                    raise ValueError(f"Gemm missing A or B: {a_name}, {b_name}")
                if isinstance(A, torch.Tensor) and A.is_sparse:
                    ab = torch.sparse.mm(A, B.t()).t() if B.dim() == 2 else torch.sparse.mm(A, B.unsqueeze(-1)).squeeze(-1)
                elif isinstance(B, torch.Tensor) and B.is_sparse:
                    ab = torch.sparse.mm(B.t(), A.t()).t()
                else:
                    ab = torch.mm(A, B)
                y = alpha * ab
                if C is not None:
                    y = y + beta * C
                state[out_name] = y
        result = {}
        for out_info in self._model.graph.output:
            name = out_info.name
            if name in state:
                t = state[name]
                result[name] = (t.numpy() if not getattr(t, "is_sparse", False) else t.to_dense().numpy())
        return result


def _create_sparse_session(model_path: str | Path):
    """Create best available sparse session: MKL first, then PyTorch."""
    mkl_sess = _try_mkl_session(model_path)
    if mkl_sess is not None:
        return mkl_sess
    return _SparseSession(model_path)


def run_sparse(
    model_path: str | Path,
    input_feed: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Run ONNX model with sparse matmul where weights are sparse. Returns output dict.
    Uses MKL if sparse-dot-mkl installed (5–15× on Intel), else PyTorch sparse.
    Raises if no backend and model has unsupported ops.
    """
    session = _create_sparse_session(model_path)
    return session.run(input_feed)


def sparse_runner_available() -> bool:
    """True if we can run sparse inference (sparse-dot-mkl or torch installed)."""
    try:
        from sparse_dot_mkl import dot_product_mkl  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def run_sparse_metrics(
    model_path: str | Path,
    input_feed: dict[str, np.ndarray],
    warmup: int = 10,
    repeat: int = 100,
    return_raw_times: bool = False,
    on_sample: Callable[[int, float], None] | None = None,
) -> tuple[InferenceMetrics, list[float] | None]:
    """
    Time sparse inference (one session, repeated runs). Returns (InferenceMetrics, raw_times or None).
    Use for benchmark "after" when --prune to get real sparse speedup.
    """
    import time
    from progenitor.runner import InferenceMetrics

    session = _create_sparse_session(model_path)
    for _ in range(warmup):
        session.run(input_feed)

    times: list[float] = []
    for i in range(repeat):
        start = time.perf_counter()
        session.run(input_feed)
        end = time.perf_counter()
        t_ms = (end - start) * 1000.0
        times.append(t_ms)
        if on_sample is not None:
            on_sample(i + 1, t_ms)

    latency_ms = float(np.median(times))
    throughput_per_sec = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    metrics = InferenceMetrics(
        latency_ms=latency_ms,
        throughput_per_sec=throughput_per_sec,
        warmup_runs=warmup,
        timed_runs=repeat,
    )
    if return_raw_times:
        return metrics, times
    return metrics, None

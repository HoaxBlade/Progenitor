"""
Native C-backed sparse MLP forward pass using Apple Accelerate SparseBLAS.

Compiles _sparse_forward.c into a .dylib, then calls it via ctypes.
Entire forward pass (all layers) runs in one C call — no Python per-node overhead.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np
from onnx import ModelProto, numpy_helper

_DYLIB_PATH = Path(__file__).parent / "_sparse_forward.dylib"
_lib = None
_WEIGHT_OPS = {"MatMul": [0, 1], "Gemm": [0, 1]}


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    if platform.system() != "Darwin":
        raise ImportError("Native Accelerate sparse is macOS-only")
    if not _DYLIB_PATH.exists():
        raise ImportError(f"Compiled library not found: {_DYLIB_PATH}. Run: cc -O3 -shared -o {_DYLIB_PATH} {_DYLIB_PATH.with_suffix('.c')} -framework Accelerate -fPIC")
    _lib = ctypes.cdll.LoadLibrary(str(_DYLIB_PATH))

    _lib.smlp_create.restype = ctypes.c_void_p
    _lib.smlp_create.argtypes = [ctypes.c_int]

    _lib.smlp_set_layer.restype = None
    _lib.smlp_set_layer.argtypes = [
        ctypes.c_void_p,                   # model
        ctypes.c_int,                      # idx
        ctypes.POINTER(ctypes.c_float),    # W_dense
        ctypes.c_int,                      # in_dim
        ctypes.c_int,                      # out_dim
        ctypes.POINTER(ctypes.c_float),    # bias (can be NULL)
        ctypes.c_int,                      # has_relu
    ]

    _lib.smlp_forward.restype = None
    _lib.smlp_forward.argtypes = [
        ctypes.c_void_p,                   # model
        ctypes.POINTER(ctypes.c_float),    # x
        ctypes.POINTER(ctypes.c_float),    # output
    ]

    _lib.smlp_destroy.restype = None
    _lib.smlp_destroy.argtypes = [ctypes.c_void_p]
    return _lib


def _weight_names(model: ModelProto) -> set[str]:
    init_names = {i.name for i in model.graph.initializer}
    names: set[str] = set()
    for node in model.graph.node:
        indices = _WEIGHT_OPS.get(node.op_type)
        if indices is None:
            continue
        for i in indices:
            if i < len(node.input) and node.input[i] in init_names:
                names.add(node.input[i])
    return names


def _execution_order(model: ModelProto) -> list:
    ready = set(inp.name for inp in model.graph.input)
    ready |= {i.name for i in model.graph.initializer}
    order, remaining = [], list(model.graph.node)
    while remaining:
        for i, n in enumerate(remaining):
            if all(inp in ready for inp in n.input):
                order.append(n)
                ready |= set(n.output)
                remaining.pop(i)
                break
        else:
            raise ValueError("Graph has cycles or missing inputs")
    return order


class NativeSparseSession:
    """
    Run pruned MLP-like ONNX models with a single C call per inference.
    Supports graphs of: MatMul + Add + Relu (repeated).
    """

    def __init__(self, model_path: str | Path) -> None:
        from progenitor.loader import load_onnx
        lib = _get_lib()
        model = load_onnx(model_path)
        self._model = model
        init_arrays = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}
        order = _execution_order(model)

        # Parse layers: each "layer" is one of:
        #   MatMul -> (Add) -> (Relu)
        #   Gemm(W, bias) -> (Relu)
        layers = []
        i = 0
        while i < len(order):
            node = order[i]
            if node.op_type == "Gemm":
                # Gemm: Y = alpha * A @ B + beta * C
                w_name = node.input[1] if node.input[1] in init_arrays else node.input[0]
                W = init_arrays[w_name]
                in_dim, out_dim = W.shape
                bias = None
                if len(node.input) > 2 and node.input[2] in init_arrays:
                    bias = init_arrays[node.input[2]]
                has_relu = 0
                if i + 1 < len(order) and order[i + 1].op_type == "Relu":
                    has_relu = 1
                    i += 1
                layers.append((W, in_dim, out_dim, bias, has_relu))
                i += 1
            elif node.op_type == "MatMul":
                w_name = None
                for inp_idx in (0, 1):
                    if node.input[inp_idx] in init_arrays and init_arrays[node.input[inp_idx]].ndim == 2:
                        w_name = node.input[inp_idx]
                        break
                if w_name is None:
                    raise ValueError("MatMul node has no weight initializer")
                W = init_arrays[w_name]
                in_dim, out_dim = W.shape
                bias = None
                has_relu = 0
                if i + 1 < len(order) and order[i + 1].op_type == "Add":
                    add_node = order[i + 1]
                    for inp in add_node.input:
                        if inp in init_arrays and init_arrays[inp].ndim == 1:
                            bias = init_arrays[inp]
                            break
                    i += 1
                    if i + 1 < len(order) and order[i + 1].op_type == "Relu":
                        has_relu = 1
                        i += 1
                elif i + 1 < len(order) and order[i + 1].op_type == "Relu":
                    has_relu = 1
                    i += 1
                layers.append((W, in_dim, out_dim, bias, has_relu))
                i += 1
            elif node.op_type == "Relu":
                i += 1
            else:
                raise ValueError(f"Unsupported op: {node.op_type}. Only MatMul/Gemm/Add/Relu supported.")

        self._n_layers = len(layers)
        self._input_dim = layers[0][1]
        self._output_dim = layers[-1][2]

        # Build C model
        self._c_model = lib.smlp_create(self._n_layers)
        for idx, (W, in_dim, out_dim, bias, has_relu) in enumerate(layers):
            W_c = np.ascontiguousarray(W, dtype=np.float32)
            W_ptr = W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if bias is not None:
                bias_c = np.ascontiguousarray(bias, dtype=np.float32)
                bias_ptr = bias_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                bias_ptr = ctypes.POINTER(ctypes.c_float)()
            lib.smlp_set_layer(self._c_model, idx, W_ptr, in_dim, out_dim, bias_ptr, has_relu)

        # Get output name
        self._output_name = model.graph.output[0].name
        self._input_name = model.graph.input[0].name

    def __del__(self):
        try:
            lib = _get_lib()
            lib.smlp_destroy(self._c_model)
        except Exception:
            pass

    def run(self, input_feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        lib = _get_lib()
        x = np.ascontiguousarray(input_feed[self._input_name].ravel(), dtype=np.float32)
        output = np.zeros(self._output_dim, dtype=np.float32)
        lib.smlp_forward(
            self._c_model,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return {self._output_name: output.reshape(1, -1)}


def native_sparse_available() -> bool:
    try:
        _get_lib()
        return True
    except Exception:
        return False

#!/usr/bin/env python3
"""
Create a larger MLP for sparse benchmark: 5–15× speedup needs big matmuls.

Run from repo root. Creates examples/large_mlp.onnx.
  python examples/export_large_mlp.py
  progenitor enhance examples/large_mlp.onnx -o examples/large_mlp_pruned.onnx --prune 0.9
  python benchmarks/run.py examples/large_mlp.onnx --prune 0.9
"""

import numpy as np
from pathlib import Path

from onnx import helper, TensorProto, numpy_helper
from onnx import save as onnx_save

HIDDEN = 1024  # increase to 4096+ for heavier matmuls


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "large_mlp.onnx"

    np.random.seed(42)
    # (1, HIDDEN) -> (HIDDEN, HIDDEN) -> (HIDDEN, HIDDEN) -> (HIDDEN, 10)
    W1 = np.random.randn(HIDDEN, HIDDEN).astype(np.float32) * 0.05
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    W2 = np.random.randn(HIDDEN, HIDDEN).astype(np.float32) * 0.05
    b2 = np.zeros(HIDDEN, dtype=np.float32)
    W3 = np.random.randn(HIDDEN, 10).astype(np.float32) * 0.05
    b3 = np.zeros(10, dtype=np.float32)

    input_x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, HIDDEN])
    output_y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    w1_init = numpy_helper.from_array(W1, "W1")
    b1_init = numpy_helper.from_array(b1, "b1")
    w2_init = numpy_helper.from_array(W2, "W2")
    b2_init = numpy_helper.from_array(b2, "b2")
    w3_init = numpy_helper.from_array(W3, "W3")
    b3_init = numpy_helper.from_array(b3, "b3")

    matmul1 = helper.make_node("MatMul", ["input", "W1"], ["m1"])
    add1 = helper.make_node("Add", ["m1", "b1"], ["a1"])
    relu1 = helper.make_node("Relu", ["a1"], ["r1"])
    matmul2 = helper.make_node("MatMul", ["r1", "W2"], ["m2"])
    add2 = helper.make_node("Add", ["m2", "b2"], ["a2"])
    relu2 = helper.make_node("Relu", ["a2"], ["r2"])
    matmul3 = helper.make_node("MatMul", ["r2", "W3"], ["m3"])
    add3 = helper.make_node("Add", ["m3", "b3"], ["output"])

    graph = helper.make_graph(
        [matmul1, add1, relu1, matmul2, add2, relu2, matmul3, add3],
        "large_mlp",
        [input_x],
        [output_y],
        initializer=[w1_init, b1_init, w2_init, b2_init, w3_init, b3_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_save(model, str(out_path))
    print(f"Created {out_path} (hidden={HIDDEN}). Prune and benchmark with --prune 0.9 for 5–15×.")


if __name__ == "__main__":
    main()

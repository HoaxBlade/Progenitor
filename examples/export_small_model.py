#!/usr/bin/env python3
"""
Create a small multi-layer ONNX model for testing Progenitor (no PyTorch required).

Uses only the 'onnx' package. Run from repo root. Creates examples/small_mlp.onnx.
"""

import numpy as np
from pathlib import Path

from onnx import helper, TensorProto, numpy_helper
from onnx import save as onnx_save


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "small_mlp.onnx"

    # Small MLP: input (1, 64) -> Linear(64, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)
    np.random.seed(42)
    W1 = np.random.randn(64, 128).astype(np.float32) * 0.1
    b1 = np.zeros(128, dtype=np.float32)
    W2 = np.random.randn(128, 64).astype(np.float32) * 0.1
    b2 = np.zeros(64, dtype=np.float32)
    W3 = np.random.randn(64, 10).astype(np.float32) * 0.1
    b3 = np.zeros(10, dtype=np.float32)

    input_x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    output_y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    w1_init = numpy_helper.from_array(W1, "W1")
    b1_init = numpy_helper.from_array(b1, "b1")
    w2_init = numpy_helper.from_array(W2, "W2")
    b2_init = numpy_helper.from_array(b2, "b2")
    w3_init = numpy_helper.from_array(W3, "W3")
    b3_init = numpy_helper.from_array(b3, "b3")

    # input -> MatMul(W1) -> Add(b1) -> Relu -> MatMul(W2) -> Add(b2) -> Relu -> MatMul(W3) -> Add(b3) -> output
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
        "small_mlp",
        [input_x],
        [output_y],
        initializer=[w1_init, b1_init, w2_init, b2_init, w3_init, b3_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_save(model, str(out_path))

    print(f"Created {out_path}")
    print("Run: progenitor enhance examples/small_mlp.onnx -o examples/small_mlp_enhanced.onnx")
    print("Then: python benchmarks/run.py examples/small_mlp.onnx --target cpu")


if __name__ == "__main__":
    main()

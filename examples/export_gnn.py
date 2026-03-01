#!/usr/bin/env python3
"""Export a minimal GNN-like ONNX for testing Progenitor GNN path.

Graph: input -> Gather -> MatMul*4 (message/readout style) -> output.
Uses only the 'onnx' package. Run from repo root. Creates examples/gnn_like.onnx.
"""

import numpy as np
from pathlib import Path

from onnx import helper, TensorProto, numpy_helper
from onnx import save as onnx_save


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "gnn_like.onnx"

    np.random.seed(42)
    # Gather1: input (1,16) -> indices (8,) axis=1 -> (1,8). So W1 must be (8, 32).
    # Gather2: r3 (1,32) -> indices (10,) axis=1 -> (1,10). So W4 must be (10, 10).
    W1 = np.random.randn(8, 32).astype(np.float32) * 0.1   # (1,8) @ (8,32) -> (1,32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.random.randn(32, 32).astype(np.float32) * 0.1
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.random.randn(32, 32).astype(np.float32) * 0.1
    b3 = np.zeros(32, dtype=np.float32)
    W4 = np.random.randn(10, 10).astype(np.float32) * 0.1  # (1,10) @ (10,10) -> (1,10)
    b4 = np.zeros(10, dtype=np.float32)

    indices1 = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    indices2 = np.arange(10, dtype=np.int64)

    input_x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16])
    output_y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    inits = [
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(b2, "b2"),
        numpy_helper.from_array(W3, "W3"),
        numpy_helper.from_array(b3, "b3"),
        numpy_helper.from_array(W4, "W4"),
        numpy_helper.from_array(b4, "b4"),
        numpy_helper.from_array(indices1, "indices1"),
        numpy_helper.from_array(indices2, "indices2"),
    ]

    nodes = [
        helper.make_node("Gather", ["input", "indices1"], ["g1"], axis=1),
        helper.make_node("MatMul", ["g1", "W1"], ["m1"]),
        helper.make_node("Add", ["m1", "b1"], ["a1"]),
        helper.make_node("Relu", ["a1"], ["r1"]),
        helper.make_node("MatMul", ["r1", "W2"], ["m2"]),
        helper.make_node("Add", ["m2", "b2"], ["a2"]),
        helper.make_node("Relu", ["a2"], ["r2"]),
        helper.make_node("MatMul", ["r2", "W3"], ["m3"]),
        helper.make_node("Add", ["m3", "b3"], ["a3"]),
        helper.make_node("Relu", ["a3"], ["r3"]),
        helper.make_node("Gather", ["r3", "indices2"], ["g2"], axis=1),
        helper.make_node("MatMul", ["g2", "W4"], ["m4"]),
        helper.make_node("Add", ["m4", "b4"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "gnn_like", [input_x], [output_y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_save(model, str(out_path))

    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("Run: progenitor enhance examples/gnn_like.onnx --target cpu --max-speed")
    print("Then: python benchmarks/run.py examples/gnn_like.onnx --target cpu --max-speed --validate --repeat 20")


if __name__ == "__main__":
    main()

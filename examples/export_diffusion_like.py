#!/usr/bin/env python3
"""Export a minimal diffusion-style ONNX (conv + attention) for testing Progenitor.

Graph: Conv*4 + MatMul*4 + Softmax + LayerNorm to satisfy is_diffusion detection.
Uses only the 'onnx' package. Run from repo root. Creates examples/diffusion_like.onnx.
"""

import numpy as np
from pathlib import Path

from onnx import helper, TensorProto, numpy_helper
from onnx import save as onnx_save


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "diffusion_like.onnx"

    np.random.seed(42)
    # Small spatial: 8x8, 4 channels -> 8 -> 8 -> 8 -> flatten -> 64 dims
    # 4 Convs: (1,4,8,8) -> (1,8,8,8) -> (1,8,8,8) -> (1,8,8,8) -> (1,8,8,8)
    c1 = np.random.randn(8, 4, 3, 3).astype(np.float32) * 0.1
    c2 = np.random.randn(8, 8, 3, 3).astype(np.float32) * 0.1
    c3 = np.random.randn(8, 8, 3, 3).astype(np.float32) * 0.1
    c4 = np.random.randn(8, 8, 3, 3).astype(np.float32) * 0.1
    # After 4 convs: (1,8,8,8) -> flatten (1, 512). Then 4 MatMuls: 512->64->64->64->10
    w1 = np.random.randn(512, 64).astype(np.float32) * 0.1
    b1 = np.zeros(64, dtype=np.float32)
    w2 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b2 = np.zeros(64, dtype=np.float32)
    w3 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b3 = np.zeros(64, dtype=np.float32)
    w4 = np.random.randn(64, 10).astype(np.float32) * 0.1
    b4 = np.zeros(10, dtype=np.float32)
    # LayerNorm: normalized shape [64]
    ln_scale = np.ones(64, dtype=np.float32)
    ln_bias = np.zeros(64, dtype=np.float32)

    input_x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
    output_y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    inits = [
        numpy_helper.from_array(c1, "c1"),
        numpy_helper.from_array(c2, "c2"),
        numpy_helper.from_array(c3, "c3"),
        numpy_helper.from_array(c4, "c4"),
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
        numpy_helper.from_array(w3, "w3"),
        numpy_helper.from_array(b3, "b3"),
        numpy_helper.from_array(w4, "w4"),
        numpy_helper.from_array(b4, "b4"),
        numpy_helper.from_array(ln_scale, "ln_scale"),
        numpy_helper.from_array(ln_bias, "ln_bias"),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "c1"], ["conv1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1"], ["r1"]),
        helper.make_node("Conv", ["r1", "c2"], ["conv2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv2"], ["r2"]),
        helper.make_node("Conv", ["r2", "c3"], ["conv3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv3"], ["r3"]),
        helper.make_node("Conv", ["r3", "c4"], ["conv4"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv4"], ["r4"]),
        helper.make_node("Flatten", ["r4"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "w1"], ["m1"]),
        helper.make_node("Add", ["m1", "b1"], ["a1"]),
        helper.make_node("Relu", ["a1"], ["att_in"]),
        helper.make_node("MatMul", ["att_in", "w2"], ["m2"]),
        helper.make_node("Softmax", ["m2"], ["sm"]),
        helper.make_node("MatMul", ["sm", "w3"], ["m3"]),
        helper.make_node("Add", ["m3", "b3"], ["a3"]),
        helper.make_node("LayerNormalization", ["a3", "ln_scale", "ln_bias"], ["ln"], axis=-1),
        helper.make_node("MatMul", ["ln", "w4"], ["m4"]),
        helper.make_node("Add", ["m4", "b4"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "diffusion_like", [input_x], [output_y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_save(model, str(out_path))

    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("Run: progenitor enhance examples/diffusion_like.onnx --target cpu --max-speed")
    print("Then: python benchmarks/run.py examples/diffusion_like.onnx --target cpu --max-speed --validate --repeat 20")


if __name__ == "__main__":
    main()

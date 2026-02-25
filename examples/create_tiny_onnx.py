#!/usr/bin/env python3
"""Create a minimal ONNX model for testing Progenitor. Run from repo root."""

import sys
from pathlib import Path

# So we can use onnx without installing progenitor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx import helper, TensorProto
from onnx import save as onnx_save

# Minimal graph: two inputs -> Add -> output
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
node = helper.make_node("Add", ["x", "y"], ["z"])
out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
graph = helper.make_graph([node], "tiny", [X, Y], [out])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

out_path = Path(__file__).parent / "tiny.onnx"
onnx_save(model, str(out_path))
print(f"Created {out_path}")

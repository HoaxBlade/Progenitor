"""ONNX graph-level optimization passes."""

from onnx import ModelProto, shape_inference


def apply_shape_inference(model: ModelProto) -> ModelProto:
    """Run ONNX shape inference. Helps runtimes and later passes."""
    return shape_inference.infer_shapes(model)

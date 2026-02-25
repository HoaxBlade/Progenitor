"""Tests for progenitor.api.enhance and pipeline."""

import tempfile
from pathlib import Path

import pytest

from onnx import helper, TensorProto
from onnx import save as onnx_save

from progenitor.api import enhance
from progenitor.loader import load_onnx


def _tiny_onnx(path: Path) -> None:
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", ["x", "y"], ["z"])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([node], "tiny", [X, Y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_save(model, str(path))


def test_enhance_compatible_onnx() -> None:
    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "model.onnx"
        _tiny_onnx(model_path)
        out_path = Path(d) / "out.onnx"

        result = enhance(model_path, "cpu", output_path=out_path)

        assert result.compatible is True
        assert result.output_path == out_path
        assert out_path.exists()
        # Load and sanity check
        loaded = load_onnx(out_path)
        assert loaded.graph.input[0].name == "x"


def test_enhance_incompatible_path() -> None:
    result = enhance("/nonexistent/model.onnx", "cpu")
    assert result.compatible is False
    assert "not found" in result.message.lower() or "incompatible" in result.message.lower()


def test_enhance_default_output_path() -> None:
    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "m.onnx"
        _tiny_onnx(model_path)

        result = enhance(model_path, "cpu")

        assert result.compatible is True
        assert result.output_path == Path(d) / "m_enhanced.onnx"
        assert result.output_path.exists()

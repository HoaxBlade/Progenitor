"""Tests for progenitor.runner metrics."""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from onnx import helper, TensorProto
from onnx import save as onnx_save

from progenitor.runner import create_random_feed, run_metrics


def _tiny_onnx(path: Path) -> None:
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", ["x", "y"], ["z"])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([node], "tiny", [X, Y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx_save(model, str(path))


def test_run_metrics_returns_metrics() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.onnx"
        _tiny_onnx(path)
        feed = {"x": np.zeros((1, 4), dtype=np.float32), "y": np.ones((1, 4), dtype=np.float32)}

        m = run_metrics(path, feed, warmup=2, repeat=5)

        assert m.latency_ms >= 0
        assert m.throughput_per_sec >= 0
        assert m.warmup_runs == 2
        assert m.timed_runs == 5


def test_create_random_feed() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.onnx"
        _tiny_onnx(path)
        sess = ort.InferenceSession(str(path), ort.SessionOptions(), providers=["CPUExecutionProvider"])
        feed = create_random_feed(sess)
        assert "x" in feed and "y" in feed
        assert feed["x"].shape == (1, 4)
        assert feed["y"].shape == (1, 4)

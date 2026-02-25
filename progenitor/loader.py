"""Load ONNX models and check compatibility."""

from pathlib import Path

import onnx
from onnx import ModelProto


def load_onnx(path: str | Path) -> ModelProto:
    """Load an ONNX model from disk. Raises if invalid or incompatible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected .onnx file, got {path.suffix}")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def save_onnx(model: ModelProto, path: str | Path) -> None:
    """Save an ONNX model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))

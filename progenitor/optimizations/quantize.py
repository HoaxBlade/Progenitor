"""Dynamic INT8 quantization for drastic CPU speedups (2–4x typical)."""

import tempfile
from pathlib import Path

from onnx import ModelProto

from progenitor.loader import save_onnx


def apply_dynamic_quantization(model: ModelProto, output_path: Path) -> None:
    """
    Quantize model to INT8 using ONNX Runtime's dynamic quantization.
    Saves the quantized model to output_path. Use for CPU inference to get large speedups.
    May reduce accuracy slightly; validate on your task.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        save_onnx(model, tmp_path)
        # Default weight_type=QuantType.QInt8; QDQ format is default and fast on CPU
        quantize_dynamic(
            tmp_path,
            str(output_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

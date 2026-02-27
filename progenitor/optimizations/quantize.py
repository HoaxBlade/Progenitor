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

class RandomCalibrationDataReader:
    """Provides random calibration data for static quantization.
    Warning: using random data for calibration may degrade accuracy vs real data."""
    def __init__(self, model_path: str, num_samples: int = 10):
        import onnxruntime as ort
        from progenitor.runner import create_random_feed
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.num_samples = num_samples
        self.iter_count = 0

    def get_next(self) -> dict | None:
        from progenitor.runner import create_random_feed
        if self.iter_count >= self.num_samples:
            return None
        self.iter_count += 1
        return create_random_feed(self.session)


def apply_static_quantization(model: ModelProto, output_path: Path) -> None:
    """
    Quantize model to INT8 using ONNX Runtime's static quantization.
    Static int8 is dramatically faster than dynamic int8 on CPUs.
    Uses random data for calibration. For production, supply actual domain data.
    """
    from onnxruntime.quantization import QuantType, quantize_static
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        save_onnx(model, tmp_path)
        
        dr = RandomCalibrationDataReader(tmp_path, num_samples=15)
        
        quantize_static(
            tmp_path,
            str(output_path),
            calibration_data_reader=dr,
            quant_format=None, # defaults to QOperator or QDQ
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

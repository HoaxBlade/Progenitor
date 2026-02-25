"""Main API: enhance a model to peak for the given target."""

from dataclasses import dataclass
from pathlib import Path

from progenitor.config import EnhanceOptions, Target
from progenitor.loader import load_onnx, save_onnx
from progenitor.optimizations.passes import apply_shape_inference


@dataclass
class EnhanceResult:
    """Result of enhancing a model."""

    input_path: Path
    output_path: Path
    target: Target
    compatible: bool
    message: str


def enhance(model_path: str | Path, target: str | Target, *, output_path: str | Path | None = None, quantize: bool = False) -> EnhanceResult:
    """
    Expose a compatible ONNX model to Progenitor; produce an optimized artifact for the target.

    - model_path: path to .onnx file
    - target: "cpu" or "cuda"
    - output_path: where to write the enhanced .onnx (default: same dir as model, name with _enhanced suffix)
    - quantize: if True, apply INT8 dynamic quantization (2–4x on CPU).

    Returns EnhanceResult with output_path and compatibility info.
    """
    model_path = Path(model_path)
    t = Target.from_id(target) if isinstance(target, str) else target
    opts = EnhanceOptions(target=t, output_path=Path(output_path) if output_path else None, quantize=quantize)

    try:
        model = load_onnx(model_path)
    except Exception as e:
        return EnhanceResult(
            input_path=model_path,
            output_path=model_path,
            target=t,
            compatible=False,
            message=f"Incompatible or invalid model: {e}",
        )

    # Graph-level passes
    model = apply_shape_inference(model)

    out = opts.output_path
    if out is None:
        suffix = "_quantized.onnx" if opts.quantize else "_enhanced.onnx"
        out = model_path.parent / f"{model_path.stem}{suffix}"
    out = Path(out)

    if opts.quantize:
        try:
            from progenitor.optimizations.quantize import apply_dynamic_quantization
            apply_dynamic_quantization(model, out)
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message="Quantized (INT8). Use the quantized model for inference; expect 2–4x speedup on CPU. Validate accuracy.",
            )
        except Exception as e:
            # Fallback: save fp32 enhanced only
            save_onnx(model, out)
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message=f"Quantization failed ({e}); saved graph-enhanced FP32 only. Run without --quantize for normal enhance.",
            )
    save_onnx(model, out)
    return EnhanceResult(
        input_path=model_path,
        output_path=out,
        target=t,
        compatible=True,
        message="Enhanced. Run with ONNX Runtime graph_optimization_level=ORT_ENABLE_ALL for peak inference.",
    )

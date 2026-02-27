"""Main API: enhance a model to peak for the given target."""

from dataclasses import dataclass
from pathlib import Path

from progenitor.config import EnhanceOptions, Target
from progenitor.loader import load_onnx, save_onnx
from progenitor.optimizations.passes import apply_shape_inference, apply_onnx_simplifier, apply_ort_offline_optimization


@dataclass
class EnhanceResult:
    """Result of enhancing a model."""

    input_path: Path
    output_path: Path
    target: Target
    compatible: bool
    message: str


def enhance(
    model_path: str | Path,
    target: str | Target,
    *,
    output_path: str | Path | None = None,
    quantize: bool = False,
    prune: float | None = None,
) -> EnhanceResult:
    """
    Expose a compatible ONNX model to Progenitor; produce an optimized artifact for the target.

    - model_path: path to .onnx file
    - target: "cpu" or "cuda"
    - output_path: where to write the enhanced .onnx (default: same dir as model, name with _enhanced suffix)
    - quantize: if True, apply INT8 dynamic quantization (2–4x on CPU).
    - prune: if set (e.g. 0.9), magnitude-based pruning to that sparsity (90% zeros); use sparse inference for 5–15×.

    Returns EnhanceResult with output_path and compatibility info.
    """
    model_path = Path(model_path)
    t = Target.from_id(target) if isinstance(target, str) else target
    opts = EnhanceOptions(
        target=t,
        output_path=Path(output_path) if output_path else None,
        quantize=quantize,
        prune=prune,
    )

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
    model = apply_onnx_simplifier(model)

    out = opts.output_path
    if out is None:
        if opts.quantize:
            suffix = "_quantized.onnx"
        elif opts.prune is not None:
            suffix = "_pruned.onnx"
        else:
            suffix = "_enhanced.onnx"
        out = model_path.parent / f"{model_path.stem}{suffix}"
    out = Path(out)

    if opts.prune is not None:
        try:
            from progenitor.optimizations.prune import apply_pruning
            apply_pruning(model, opts.prune)
        except Exception as e:
            save_onnx(model, out)
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message=f"Pruning failed ({e}); saved graph-enhanced only.",
            )
        save_onnx(model, out)
        return EnhanceResult(
            input_path=model_path,
            output_path=out,
            target=t,
            compatible=True,
            message=f"Pruned to {opts.prune:.0%} sparsity. Same graph. For 5–15× speedup you need a sparse backend (ORT runs dense by default). Validate accuracy.",
        )

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
    # Standard enhance: bake in ORT optimizations offline
    try:
        apply_ort_offline_optimization(model, out, t)
    except Exception as e:
        save_onnx(model, out)
        return EnhanceResult(
            input_path=model_path,
            output_path=out,
            target=t,
            compatible=True,
            message=f"Enhanced (graph passes only, offline ORT fusion failed: {e}). Run with ONNX Runtime graph_optimization_level=ORT_ENABLE_ALL.",
        )
    return EnhanceResult(
        input_path=model_path,
        output_path=out,
        target=t,
        compatible=True,
        message="Enhanced. Run with ONNX Runtime graph_optimization_level=ORT_ENABLE_ALL for peak inference.",
    )

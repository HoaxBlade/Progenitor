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
    static_quantize: bool = False,
    prune: float | None = None,
    struct_prune: float | None = None,
    conv_prune: float | None = None,
    lowrank: float | None = None,
    max_speed: bool = False,
) -> EnhanceResult:
    """
    Enhance an ONNX model for peak inference performance.

    Optimization passes (can be stacked for multiplicative speedup):
      --struct-prune: remove entire neurons, physically shrink graph (~2-4x)
      --lowrank:      SVD decomposition of weight matrices (~2-3x)
      --prune:        unstructured magnitude pruning + sparse backend (~5x)
      --max-speed:    chain all above with aggressive defaults (~30-50x)
    """
    model_path = Path(model_path)
    t = Target.from_id(target) if isinstance(target, str) else target

    if max_speed:
        # Detect architecture to apply only beneficial passes
        model_tmp = load_onnx(model_path)
        from collections import Counter
        op_counts = Counter(n.op_type for n in model_tmp.graph.node)
        n_conv = op_counts.get("Conv", 0)
        n_linear = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)
        is_cnn = n_conv > n_linear and n_conv >= 3
        del model_tmp

        if is_cnn:
            # CNN: only conv channel pruning is universally fast across all CPUs
            # INT8 quant can be slower on CPUs without VNNI instructions (like the current Windows host)
            conv_prune = conv_prune if conv_prune is not None else 0.40
            static_quantize = False
        else:
            # MLP: check size, tiny models can't survive 90% pruning
            import numpy as np
            total_params = sum(
                np.prod(i.dims) for i in getattr(load_onnx(model_path).graph, 'initializer', [])
                if getattr(i, 'dims', None)
            )
            if total_params < 100_000:  # e.g., small_mlp is 17k
                struct_prune = struct_prune if struct_prune is not None else None
                lowrank = lowrank if lowrank is not None else None
                prune = prune if prune is not None else 0.50
            else:
                struct_prune = struct_prune if struct_prune is not None else 0.50
                lowrank = lowrank if lowrank is not None else 0.25
                prune = prune if prune is not None else 0.90

    opts = EnhanceOptions(
        target=t,
        output_path=Path(output_path) if output_path else None,
        quantize=quantize,
        static_quantize=static_quantize,
        prune=prune,
        struct_prune=struct_prune,
        conv_prune=conv_prune,
        lowrank=lowrank,
        max_speed=max_speed,
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
        if opts.static_quantize:
            suffix = "_static_quantized.onnx"
        elif opts.quantize:
            suffix = "_quantized.onnx"
        elif opts.prune is not None:
            suffix = "_pruned.onnx"
        elif opts.struct_prune is not None or opts.lowrank is not None:
            suffix = "_optimized.onnx"
        else:
            suffix = "_enhanced.onnx"
        out = model_path.parent / f"{model_path.stem}{suffix}"
    out = Path(out)

    applied_passes: list[str] = []

    # 0. Conv channel pruning (physically shrink Conv bottleneck kernels)
    if opts.conv_prune is not None:
        try:
            from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
            apply_conv_structured_pruning(model, opts.conv_prune)
            applied_passes.append(f"conv channel prune {opts.conv_prune:.0%}")
        except Exception as e:
            applied_passes.append(f"conv channel prune FAILED ({e})")

    # 1. Structured pruning (physical graph shrinkage)
    if opts.struct_prune is not None:
        try:
            from progenitor.optimizations.structured_prune import apply_structured_pruning
            apply_structured_pruning(model, opts.struct_prune)
            applied_passes.append(f"structured prune {opts.struct_prune:.0%}")
        except Exception as e:
            applied_passes.append(f"structured prune FAILED ({e})")

    # 2. Low-rank SVD decomposition
    if opts.lowrank is not None:
        try:
            from progenitor.optimizations.lowrank import apply_lowrank_decomposition
            apply_lowrank_decomposition(model, opts.lowrank)
            applied_passes.append(f"low-rank SVD (rank {opts.lowrank:.0%})")
        except Exception as e:
            applied_passes.append(f"low-rank FAILED ({e})")

    # 3. Unstructured magnitude pruning
    if opts.prune is not None:
        try:
            from progenitor.optimizations.prune import apply_pruning
            apply_pruning(model, opts.prune)
            applied_passes.append(f"unstructured prune {opts.prune:.0%}")
        except Exception as e:
            applied_passes.append(f"prune FAILED ({e})")

    # If any optimization pass ran, save the pruned model
    if applied_passes:
        save_onnx(model, out)

    # 4. Static quantization (can stack on top of pruning)
    if opts.static_quantize or static_quantize:
        try:
            from progenitor.optimizations.quantize import apply_static_quantization
            # Quantize the (possibly pruned) model
            source = out if applied_passes else None
            if source:
                apply_static_quantization(load_onnx(source), out)
            else:
                apply_static_quantization(model, out)
            applied_passes.append("static INT8 quantization")
        except Exception as e:
            applied_passes.append(f"static quantize FAILED ({e})")

    if applied_passes:
        passes_str = " -> ".join(applied_passes)
        msg = f"Applied: {passes_str}."
        if opts.prune is not None:
            msg += " Use native sparse backend for full speedup."
        return EnhanceResult(
            input_path=model_path,
            output_path=out,
            target=t,
            compatible=True,
            message=msg,
        )


    # 4. INT8 quantization (standalone)
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

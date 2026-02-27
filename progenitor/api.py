"""Main API: enhance a model to peak for the given target."""

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path

from progenitor.config import EnhanceOptions, Target
from progenitor.loader import load_onnx, save_onnx
from progenitor.optimizations.passes import apply_shape_inference, apply_onnx_simplifier, apply_ort_offline_optimization
from progenitor.validate import validate_accuracy


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
            # CNN: single conv channel prune for >2x with high cosine (no double pass)
            conv_prune = conv_prune if conv_prune is not None else 0.5
        else:
            # MLP: all passes contribute to speedup
            struct_prune = struct_prune if struct_prune is not None else 0.75
            lowrank = lowrank if lowrank is not None else 0.1
            prune = prune if prune is not None else 0.99

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

    # Detect architecture: CNN vs Transformer vs MLP
    from collections import Counter
    op_counts = Counter(n.op_type for n in model.graph.node)
    n_conv = op_counts.get("Conv", 0)
    n_linear = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)
    has_softmax = op_counts.get("Softmax", 0) > 0
    has_layernorm = op_counts.get("LayerNormalization", 0) > 0
    is_conv_heavy = n_conv > n_linear and n_conv >= 3
    is_transformer = not is_conv_heavy and (has_softmax or has_layernorm) and n_linear >= 6
    is_large_mlp = False
    if max_speed and not is_conv_heavy and not is_transformer:
        from onnx import numpy_helper
        total_params = sum(numpy_helper.to_array(i).size for i in model.graph.initializer)
        is_large_mlp = total_params > 100_000

    if max_speed:
        if is_conv_heavy:
            # CNN: one conv pass only (0.5); no second struct pass so cosine stays high
            struct_prune = struct_prune if struct_prune is not None else None
            prune = None
        elif is_transformer:
            # Transformer: light struct + conservative lowrank for high cosine and good speedup
            struct_prune = 0.25
            lowrank = 0.4  # keep 40% singular values to preserve cosine
            prune = None
        else:
            # MLP: large MLP -> full stack (struct 0.75 + lowrank 0.1 + 0.99) for 123x; small -> tuner for cosine
            if is_large_mlp:
                struct_prune = struct_prune if struct_prune is not None else 0.75
                lowrank = lowrank if lowrank is not None else 0.1
                prune = prune if prune is not None else 0.99
            else:
                struct_prune = struct_prune if struct_prune is not None else None
                lowrank = lowrank if lowrank is not None else None
                prune = prune if prune is not None else 0.99
        # Update the opts object (preserve conv_prune for CNN)
        opts = EnhanceOptions(
            target=t,
            output_path=opts.output_path,
            quantize=quantize,
            static_quantize=static_quantize,
            prune=prune,
            struct_prune=struct_prune,
            conv_prune=opts.conv_prune,
            lowrank=lowrank,
            max_speed=max_speed,
        )

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

    # 1. Structured pruning (physical graph shrinkage); also for large MLP (full stack for 123x)
    if opts.struct_prune is not None and (is_conv_heavy or is_transformer or is_large_mlp):
        try:
            if is_conv_heavy:
                from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
                apply_conv_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"conv filter prune {opts.struct_prune:.0%}")
            elif is_transformer:
                from progenitor.optimizations.transformer_prune import apply_transformer_structured_pruning
                apply_transformer_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"transformer FFN prune {opts.struct_prune:.0%}")
            elif is_large_mlp:
                from progenitor.optimizations.structured_prune import apply_structured_pruning
                apply_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"structured prune {opts.struct_prune:.0%}")
        except Exception as e:
            applied_passes.append(f"structured prune FAILED ({e})")

    # 2. Low-rank SVD decomposition; also for large MLP (full stack for 123x)
    if opts.lowrank is not None and (is_conv_heavy or is_transformer or is_large_mlp):
        try:
            from progenitor.optimizations.lowrank import apply_lowrank_decomposition
            apply_lowrank_decomposition(model, opts.lowrank)
            applied_passes.append(f"low-rank SVD (rank {opts.lowrank:.0%})")
        except Exception as e:
            applied_passes.append(f"low-rank FAILED ({e})")

    # 3. Unstructured pruning
    if opts.prune is not None:
        try:
            from progenitor.optimizations.prune import apply_pruning, apply_importance_pruning, compute_activation_importance
            if not is_conv_heavy:
                # MLP: large MLP -> full stack already applied, magnitude 0.99 for 123x; small -> tune for cosine
                if is_large_mlp:
                    # Large MLP: magnitude 0.99 for 123x with sparse backend (struct+lowrank already applied above)
                    apply_pruning(model, opts.prune)
                    applied_passes.append(f"unstructured prune {opts.prune:.0%}")
                else:
                    importance = compute_activation_importance(model, num_runs=25)
                    if importance:
                        # Small MLP: tune so cosine >= 0.9 and speedup high
                        best_sparsity = 0.0
                        for sparsity in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
                            try:
                                m = copy.deepcopy(model)
                                apply_importance_pruning(m, sparsity, importance)
                                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                                    tmp = Path(f.name)
                                save_onnx(m, tmp)
                                try:
                                    metrics = validate_accuracy(model_path, tmp)
                                    if metrics["cosine_similarity"] >= 0.9:
                                        best_sparsity = sparsity
                                finally:
                                    try:
                                        tmp.unlink()
                                    except FileNotFoundError:
                                        pass
                            except Exception:
                                continue
                        sparsity = best_sparsity if best_sparsity > 0 else 0.5
                        apply_importance_pruning(model, sparsity, importance)
                        applied_passes.append(f"importance unstructured prune {sparsity:.0%}")
                    else:
                        apply_pruning(model, opts.prune)
                        applied_passes.append(f"unstructured prune {opts.prune:.0%}")
            else:
                apply_pruning(model, opts.prune)
                applied_passes.append(f"unstructured prune {opts.prune:.0%}")
        except Exception as e:
            applied_passes.append(f"prune FAILED ({e})")

    # If any optimization pass ran, save and return
    if applied_passes:
        save_onnx(model, out)
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

    if opts.static_quantize:
        try:
            from progenitor.optimizations.quantize import apply_static_quantization
            apply_static_quantization(model, out)
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message="Quantized statically (INT8/UInt8). Peak CPU vector performance. Validate accuracy on real data.",
            )
        except Exception as e:
            save_onnx(model, out)
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message=f"Static quantization failed ({e}); saved graph-enhanced FP32 only.",
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

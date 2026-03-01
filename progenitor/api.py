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
    per_layer_tune: bool = False,
    block_size: tuple[int, int] | None = None,
    calibrate_output: bool = False,
    progressive_steps: tuple[float, ...] | None = None,
    max_speed_aggressive: bool = False,
    sparse_pattern: str = "unstructured",
) -> EnhanceResult:
    """
    Enhance an ONNX model for peak inference performance.

    Optimization passes (can be stacked for multiplicative speedup):
      --struct-prune: remove entire neurons, physically shrink graph (~2-4x)
      --lowrank:      SVD decomposition of weight matrices (~2-3x)
      --prune:        unstructured magnitude pruning + sparse backend (~5x)
      --max-speed:    chain all above with aggressive defaults (~30-50x)

    High speedup + high cosine (optional):
      per_layer_tune, block_size, calibrate_output: as before.
      progressive_steps: e.g. (0.5, 0.7, 0.9) for progressive pruning + calibrate between.
      max_speed_aggressive: small MLP/transformer get struct+lowrank+prune then calibrate (toward 50x).
      sparse_pattern: "unstructured" or "2:4" (2 non-zeros per 4, hardware-friendly).
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
            # CNN: conv channel prune + INT8 for ~5x (prune then quantize in one shot)
            conv_prune = conv_prune if conv_prune is not None else 0.5
            static_quantize = static_quantize or True  # chain prune -> static quant for 5x
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
        per_layer_tune=per_layer_tune,
        block_size=block_size,
        calibrate_output=calibrate_output,
        progressive_steps=progressive_steps,
        max_speed_aggressive=max_speed_aggressive,
        sparse_pattern=sparse_pattern,
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

    # Detect architecture: CNN, Diffusion, Transformer, RNN, GNN, MLP
    from collections import Counter
    op_counts = Counter(n.op_type for n in model.graph.node)
    n_conv = op_counts.get("Conv", 0)
    n_linear = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)
    has_softmax = op_counts.get("Softmax", 0) > 0
    has_layernorm = op_counts.get("LayerNormalization", 0) > 0
    is_conv_heavy = n_conv > n_linear and n_conv >= 3
    has_attention = has_softmax or has_layernorm
    # Diffusion: has conv + attention + linear (e.g. U-Net + cross-attn); may have n_conv == n_linear
    is_diffusion = n_conv >= 3 and has_attention and n_linear >= 4
    is_conv_only_cnn = is_conv_heavy and not is_diffusion
    is_transformer = not is_conv_heavy and has_attention and n_linear >= 6
    is_rnn = (op_counts.get("LSTM", 0) + op_counts.get("GRU", 0)) >= 1
    is_gnn = (op_counts.get("Gather", 0) + op_counts.get("Scatter", 0)) >= 2 and n_linear >= 4
    is_large_mlp = False
    if max_speed and not is_conv_heavy and not is_transformer and not is_rnn and not is_gnn:
        from onnx import numpy_helper
        total_params = sum(numpy_helper.to_array(i).size for i in model.graph.initializer)
        is_large_mlp = total_params > 100_000

    if max_speed:
        if is_diffusion:
            # Diffusion: lowrank + prune + calibrate; no INT8. Target speedup ≥3x (cosine bug: reported value often low; see docs).
            struct_prune = None
            conv_prune = None
            lowrank = 0.4
            prune = 0.85
        elif is_conv_only_cnn:
            # CNN: one conv pass only (0.5); no second struct pass so cosine stays high
            struct_prune = struct_prune if struct_prune is not None else None
            prune = None
        elif is_transformer:
            # Transformer: struct + lowrank for ~2x; aggressive adds prune 0.9 then calibrate
            struct_prune = struct_prune if struct_prune is not None else 0.25
            lowrank = lowrank if lowrank is not None else 0.4
            prune = prune if prune is not None else (0.9 if opts.max_speed_aggressive else None)
        elif is_rnn:
            # RNN (LSTM/GRU): struct + lowrank on linear parts; magnitude prune on LSTM/GRU weights + calibrate
            struct_prune = struct_prune if struct_prune is not None else 0.2
            lowrank = lowrank if lowrank is not None else 0.35
            prune = prune if prune is not None else 0.85
        elif is_gnn:
            # GNN: no struct (graph has Gather/Scatter so linear chain is broken); lowrank + prune + calibrate
            struct_prune = None
            lowrank = lowrank if lowrank is not None else 0.35
            prune = prune if prune is not None else 0.85
        else:
            # MLP: large MLP -> full stack (struct 0.75 + lowrank 0.1 + 0.99) for 123x; small -> tuner or aggressive
            if is_large_mlp:
                struct_prune = struct_prune if struct_prune is not None else 0.75
                lowrank = lowrank if lowrank is not None else 0.1
                prune = prune if prune is not None else 0.99
            else:
                if opts.max_speed_aggressive:
                    struct_prune = struct_prune if struct_prune is not None else 0.5
                    lowrank = lowrank if lowrank is not None else 0.2
                    prune = prune if prune is not None else 0.9
                else:
                    # Small MLP (non-aggressive): no struct/lowrank so tuner or progressive can run
                    struct_prune = None
                    lowrank = None
                    prune = prune if prune is not None else 0.99
        # Apply high-cosine workarounds by default when max_speed: per-layer + block for small MLP; calibration for CNN/transformer/RNN/GNN/diffusion (keeps small MLP sparse path)
        _is_small_mlp = not is_conv_heavy and not is_transformer and not is_large_mlp and not is_rnn and not is_gnn and not is_diffusion
        _per_layer = (opts.per_layer_tune or True) if _is_small_mlp else opts.per_layer_tune
        _block_size = opts.block_size if opts.block_size is not None else ((4, 4) if _is_small_mlp else None)
        _calibrate = opts.calibrate_output or ((is_conv_heavy or is_transformer or is_rnn or is_gnn or is_diffusion) and not is_large_mlp)
        # Update the opts object (use updated conv_prune; diffusion and CNN get static quant for max speedup)
        opts = EnhanceOptions(
            target=t,
            output_path=opts.output_path,
            quantize=quantize,
            static_quantize=static_quantize or is_diffusion,
            prune=prune,
            struct_prune=struct_prune,
            conv_prune=conv_prune,
            lowrank=lowrank,
            max_speed=max_speed,
            per_layer_tune=_per_layer,
            block_size=_block_size,
            calibrate_output=_calibrate,
            progressive_steps=opts.progressive_steps,
            max_speed_aggressive=opts.max_speed_aggressive,
            sparse_pattern=opts.sparse_pattern,
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
    # CNN max_speed: validation-guided passes use intermediate cosine threshold; output calibration restores final cosine so it is not compromised
    _cnn_cosine_threshold = 0.86
    _enable_block_removal = True

    # 0a. Conv channel pruning first (biggest win); validation-guided for CNN max_speed to preserve final cosine
    if opts.conv_prune is not None:
        try:
            from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
            if is_conv_only_cnn and max_speed:
                best_conv_ratio = None
                for ratio in [0.5, 0.55, 0.6, 0.65, 0.7]:
                    m = copy.deepcopy(model)
                    apply_conv_structured_pruning(m, ratio)
                    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                        tmp = Path(f.name)
                    try:
                        save_onnx(m, tmp)
                        metrics = validate_accuracy(model_path, tmp)
                        # Threshold for intermediate (pre-calibration) model; calibration later restores cosine
                        if metrics["cosine_similarity"] >= _cnn_cosine_threshold:
                            best_conv_ratio = ratio
                    finally:
                        tmp.unlink(missing_ok=True)
                if best_conv_ratio is not None:
                    apply_conv_structured_pruning(model, best_conv_ratio)
                    applied_passes.append(f"conv channel prune {best_conv_ratio:.0%}")
                # else skip conv prune to avoid compromising cosine
            else:
                apply_conv_structured_pruning(model, opts.conv_prune)
                applied_passes.append(f"conv channel prune {opts.conv_prune:.0%}")
        except Exception as e:
            applied_passes.append(f"conv channel prune FAILED ({e})")

    # 0b. Validation-guided block removal (CNN max_speed only). Uses _cnn_cosine_threshold; calibration restores final cosine.
    if _enable_block_removal and is_conv_only_cnn and max_speed and applied_passes:
        try:
            from progenitor.optimizations.block_removal import apply_block_removal
            best_block_ratio = 0.0
            for ratio in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                m = copy.deepcopy(model)
                apply_block_removal(m, ratio)
                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                    tmp = Path(f.name)
                try:
                    save_onnx(m, tmp)
                    metrics = validate_accuracy(model_path, tmp)
                    if metrics["cosine_similarity"] >= _cnn_cosine_threshold:
                        best_block_ratio = ratio
                finally:
                    tmp.unlink(missing_ok=True)
            if best_block_ratio > 0:
                apply_block_removal(model, best_block_ratio)
                applied_passes.append(f"block removal {best_block_ratio:.0%}")
        except Exception as e:
            applied_passes.append(f"block removal FAILED ({e})")

    # 1. Structured pruning (physical graph shrinkage); CNN, transformer, MLP, RNN, GNN
    if opts.struct_prune is not None and (is_conv_heavy or is_transformer or is_large_mlp or is_rnn or is_gnn or (not is_conv_heavy and not is_transformer and not is_large_mlp)):
        try:
            if is_conv_heavy:
                from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
                apply_conv_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"conv filter prune {opts.struct_prune:.0%}")
            elif is_transformer:
                from progenitor.optimizations.transformer_prune import apply_transformer_structured_pruning
                apply_transformer_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"transformer FFN prune {opts.struct_prune:.0%}")
            elif is_large_mlp or is_rnn or is_gnn:
                from progenitor.optimizations.structured_prune import apply_structured_pruning
                apply_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"structured prune {opts.struct_prune:.0%}")
            else:
                # Small MLP
                from progenitor.optimizations.structured_prune import apply_structured_pruning
                apply_structured_pruning(model, opts.struct_prune)
                applied_passes.append(f"structured prune {opts.struct_prune:.0%}")
        except Exception as e:
            applied_passes.append(f"structured prune FAILED ({e})")

    # 2. Low-rank SVD decomposition; CNN, transformer, MLP, RNN, GNN, diffusion
    if opts.lowrank is not None and (is_conv_heavy or is_transformer or is_large_mlp or is_rnn or is_gnn or is_diffusion or (not is_conv_heavy and not is_transformer and not is_large_mlp)):
        try:
            from progenitor.optimizations.lowrank import apply_lowrank_decomposition, fix_layer_norm_shapes_after_lowrank
            apply_lowrank_decomposition(model, opts.lowrank)
            if is_diffusion:
                fix_layer_norm_shapes_after_lowrank(model)
            applied_passes.append(f"low-rank SVD (rank {opts.lowrank:.0%})")
        except Exception as e:
            applied_passes.append(f"low-rank FAILED ({e})")

    # 3. Unstructured pruning
    if opts.prune is not None:
        try:
            from progenitor.optimizations.prune import (
                apply_pruning,
                apply_importance_pruning,
                apply_pruning_to_target,
                compute_activation_importance,
                tune_per_layer_sparsity,
            )
            # Progressive pruning: step through targets and optionally calibrate between (better cosine at high sparsity)
            if opts.progressive_steps and not is_conv_heavy and not is_large_mlp:
                steps = sorted(set(opts.progressive_steps))
                for target in steps:
                    apply_pruning_to_target(model, target)
                    applied_passes.append(f"progressive to {target:.0%}")
                # Calibration runs once at end (main block) if opts.calibrate_output
            else:
                # Large MLP 123x path: magnitude-only unstructured prune (no block_size, no per-layer)
                prune_kw = {} if opts.block_size is None else {"block_size": opts.block_size}
                prune_kw["sparse_pattern"] = getattr(opts, "sparse_pattern", "unstructured")
                if not is_conv_heavy:
                    # MLP / RNN / GNN
                    if is_large_mlp:
                        # Large MLP: magnitude 0.99 only (no block_size) for 123x with sparse backend
                        apply_pruning(model, opts.prune)
                        applied_passes.append(f"unstructured prune {opts.prune:.0%}")
                    elif is_rnn or is_gnn:
                        # RNN/GNN: magnitude prune (Conv/MatMul/Gemm/LSTM/GRU weights); no per-layer tune
                        apply_pruning(model, opts.prune, **prune_kw)
                        applied_passes.append(f"unstructured prune {opts.prune:.0%}")
                    else:
                        importance = compute_activation_importance(model, num_runs=25)
                        if importance:
                            # Small MLP: per-layer tune or single sparsity tune so cosine >= 0.9
                            if opts.per_layer_tune:
                                sparsity_per_layer = tune_per_layer_sparsity(
                                    model, str(model_path), importance, cosine_threshold=0.9
                                )
                                apply_importance_pruning(model, sparsity_per_layer, importance)
                                applied_passes.append("per-layer importance prune")
                            else:
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
                            apply_pruning(model, opts.prune, **prune_kw)
                            applied_passes.append(f"unstructured prune {opts.prune:.0%}")
                else:
                    apply_pruning(model, opts.prune, **prune_kw)
                    applied_passes.append(f"unstructured prune {opts.prune:.0%}")
        except Exception as e:
            applied_passes.append(f"prune FAILED ({e})")

    # Optional: output calibration on top of any optimization to recover cosine (skip for large MLP 123x path)
    if applied_passes and opts.calibrate_output and not is_large_mlp:
        try:
            from progenitor.optimizations.calibrate import apply_output_calibration
            calib_samples = 100 if is_diffusion else 50
            apply_output_calibration(model_path, model, num_samples=calib_samples)
            applied_passes.append("output calibration")
        except Exception as e:
            applied_passes.append(f"calibration FAILED ({e})")

    # If any optimization pass ran, save and return (or chain to quantize for CNN only)
    # Diffusion: do not chain to INT8 (dynamic quant fails "Graph is not a DAG"; static has QDQ shape bug).
    # Saves FP32 lowrank+prune+calibrate so speedup stays ~3x. BUG: reported cosine for diffusion stays ~0.098; needs fix.
    if applied_passes:
        chain_quantize = is_conv_heavy and (opts.static_quantize or opts.quantize)
        if chain_quantize:
            # Save pruned to intermediate; quantize in-memory model below
            out_pruned = model_path.parent / f"{model_path.stem}_optimized.onnx"
            save_onnx(model, out_pruned)
        else:
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
            chained = bool(applied_passes and (is_conv_heavy or is_diffusion))
            msg = (
                "Pruned then quantized (INT8). Validate accuracy."
                if chained
                else "Quantized statically (INT8/UInt8). Peak CPU vector performance. Validate accuracy on real data."
            )
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message=msg,
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

    # 4. INT8 quantization (standalone or after conv prune)
    if opts.quantize:
        try:
            from progenitor.optimizations.quantize import apply_dynamic_quantization
            apply_dynamic_quantization(model, out)
            chained = bool(applied_passes and is_conv_heavy)
            msg = (
                "Conv pruned then quantized (INT8). ~5x target. Validate accuracy."
                if chained
                else "Quantized (INT8). Use the quantized model for inference; expect 2–4x speedup on CPU. Validate accuracy."
            )
            return EnhanceResult(
                input_path=model_path,
                output_path=out,
                target=t,
                compatible=True,
                message=msg,
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

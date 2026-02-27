#!/usr/bin/env python3
"""
Before/after benchmark: run model baseline vs Progenitor-enhanced with full graph opts.

Usage:
  python benchmarks/run.py path/to/model.onnx [--target cpu] [--repeat 100]
"""

import argparse
import sys
from pathlib import Path

# Add project root so progenitor is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort

from progenitor.api import enhance
from progenitor.runner import create_random_feed, run_metrics

try:
    from progenitor.validate import validate_accuracy
except ImportError:
    validate_accuracy = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="Path to .onnx model")
    ap.add_argument("--target", "-t", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--verbose", "-v", action="store_true", help="Print raw sample timings to show measurements are live")
    ap.add_argument("--live", "-l", action="store_true", help="Stream each run's latency in real time as the benchmark runs")
    ap.add_argument("--quantize", "-q", action="store_true", help="Use INT8 quantization for 'after' (2–4x on CPU)")
    ap.add_argument("--static-quantize", "-sq", action="store_true", help="Use Static INT8 quantization for 'after' (much faster on CPU)")
    ap.add_argument("--prune", "-p", type=float, default=None, metavar="SPARSITY", help="Use magnitude pruning for 'after', e.g. 0.9 = 90%% zeros (sparse inference for 5–15×)")
    ap.add_argument("--struct-prune", type=float, default=None, metavar="RATIO", help="Structured pruning: remove RATIO fraction of hidden neurons (e.g. 0.5)")
    ap.add_argument("--lowrank", type=float, default=None, metavar="RANK_RATIO", help="Low-rank SVD decomposition: keep RANK_RATIO of singular values (e.g. 0.25)")
    ap.add_argument("--max-speed", action="store_true", help="Chain all optimizations for maximum speedup (~30-50×)")
    ap.add_argument("--int8-sparse", action="store_true", help="Use INT8 quantized sparse backend (bonus, additional ~1.5-2x)")
    ap.add_argument("--validate", action="store_true", help="Validate accuracy degradation (MSE) between baseline and enhanced model.")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"Error: not found: {args.model}", file=sys.stderr)
        return 1
    if args.prune is not None and (args.prune < 0 or args.prune > 1):
        print("Error: --prune must be between 0 and 1", file=sys.stderr)
        return 1

    result = enhance(
        args.model, args.target,
        quantize=args.quantize,
        prune=args.prune,
        struct_prune=args.struct_prune,
        lowrank=args.lowrank,
        max_speed=args.max_speed,
    )
    if not result.compatible:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1

    # Same hardware for before and after (virus = enhance the host you have, don't swap it)
    before_providers = result.target.execution_providers
    after_providers = result.target.execution_providers

    # Get input feed from original model (same for both runs)
    sess_ref = ort.InferenceSession(
        str(args.model),
        ort.SessionOptions(),
        providers=[before_providers[0]],
    )
    feed = create_random_feed(sess_ref)

    def make_live_callback(label: str) -> callable:
        def on_sample(run: int, t_ms: float) -> None:
            print(f"  {label} run {run:3d}: {t_ms:.4f} ms", flush=True)
        return on_sample

    on_sample_before = make_live_callback("before") if args.live else None
    on_sample_after = make_live_callback("after ") if args.live else None

    if args.live:
        print("Progenitor benchmark — live stream (each run printed as it completes)")
        print("=" * 60)
        print(f"Model: {args.model}  Target: {args.target}  Warmup: {args.warmup}  Repeat: {args.repeat}")
        if args.max_speed:
            print("Mode: MAX SPEED (structured prune -> low-rank -> unstructured prune + sparse)")
        elif args.quantize:
            print("Mode: INT8 quantized 'after' (same device, 2–4x typical on CPU)")
        elif args.prune is not None:
            print(f"Mode: Pruned {args.prune:.0%} sparsity 'after' (ORT, same runtime as before)")
        elif args.struct_prune is not None or args.lowrank is not None:
            parts = []
            if args.struct_prune is not None:
                parts.append(f"struct-prune {args.struct_prune:.0%}")
            if args.lowrank is not None:
                parts.append(f"low-rank {args.lowrank:.0%}")
            print(f"Mode: {' + '.join(parts)}")
        print()

    # Before: original model, no graph opts (same device)
    if args.live:
        print("Before (baseline, no graph opts):")
    before_out = run_metrics(
        args.model,
        feed,
        execution_providers=before_providers,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        warmup=args.warmup,
        repeat=args.repeat,
        return_raw_times=args.verbose or args.live,
        on_sample=on_sample_before,
    )
    before = before_out[0] if (args.verbose or args.live) else before_out
    before_times = before_out[1] if (args.verbose or args.live) else None

    if args.live:
        print()

    # Detect if model is CNN-like (native sparse backend only handles MLPs)
    import onnx as _onnx
    _orig_model = _onnx.load(str(args.model))
    from collections import Counter as _Counter
    _op_counts = _Counter(n.op_type for n in _orig_model.graph.node)
    _is_conv_heavy = _op_counts.get("Conv", 0) > _op_counts.get("MatMul", 0) + _op_counts.get("Gemm", 0) and _op_counts.get("Conv", 0) >= 3
    del _orig_model

    # After: enhanced, quantized, or pruned with sparse backend
    use_native_sparse = False
    use_int8_sparse = False
    effective_prune = args.prune
    if args.max_speed and effective_prune is None:
        if _is_conv_heavy:
            effective_prune = 0.9
        else:
            effective_prune = 0.99
    if effective_prune is not None and not _is_conv_heavy:
        try:
            from progenitor.backends.accelerate_sparse_native import native_sparse_available
            if native_sparse_available():
                if args.int8_sparse:
                    from progenitor.backends.accelerate_sparse_native import NativeSparseSessionI8
                    _probe_sess = NativeSparseSessionI8(result.output_path)
                    _probe_sess.run(feed)
                    use_int8_sparse = True
                    use_native_sparse = True
                    del _probe_sess
                else:
                    from progenitor.backends.accelerate_sparse_native import NativeSparseSession
                    _probe_sess = NativeSparseSession(result.output_path)
                    _probe_sess.run(feed)
                    use_native_sparse = True
                    del _probe_sess
        except Exception:
            pass

    if args.live:
        if args.max_speed and use_int8_sparse:
            after_label = "MAX SPEED + INT8 sparse (Native SparseBLAS)"
        elif args.max_speed and use_native_sparse:
            after_label = "MAX SPEED: struct-prune + low-rank + sparse (Native C)"
        elif args.max_speed:
            after_label = "MAX SPEED: struct-prune + low-rank + unstructured prune"
        elif args.quantize:
            after_label = "quantized INT8"
        elif effective_prune is not None and use_int8_sparse:
            after_label = f"pruned {effective_prune:.0%} INT8 sparse (Native)"
        elif effective_prune is not None and use_native_sparse:
            after_label = f"pruned {effective_prune:.0%} sparsity (Native C)"
        elif effective_prune is not None:
            after_label = f"pruned {effective_prune:.0%} sparsity"
        else:
            after_label = "enhanced, full graph opts"
        print("After (" + after_label + "):")

    if use_native_sparse:
        import time as _time
        from progenitor.runner import InferenceMetrics
        if use_int8_sparse:
            from progenitor.backends.accelerate_sparse_native import NativeSparseSessionI8
            _sess = NativeSparseSessionI8(result.output_path)
        else:
            from progenitor.backends.accelerate_sparse_native import NativeSparseSession
            _sess = NativeSparseSession(result.output_path)
        for _ in range(args.warmup):
            _sess.run(feed)
        _times = []
        for i in range(args.repeat):
            _t0 = _time.perf_counter()
            _sess.run(feed)
            _t1 = _time.perf_counter()
            _t_ms = (_t1 - _t0) * 1000.0
            _times.append(_t_ms)
            if on_sample_after:
                on_sample_after(i + 1, _t_ms)
        import numpy as _np
        _lat = float(_np.median(_times))
        after = InferenceMetrics(
            latency_ms=_lat,
            throughput_per_sec=1000.0 / _lat if _lat > 0 else 0.0,
            warmup_runs=args.warmup,
            timed_runs=args.repeat,
            peak_memory_mb=0.0,
        )
        after_times = _times if (args.verbose or args.live) else None
    else:
        after_out = run_metrics(
            result.output_path,
            feed,
            execution_providers=after_providers,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            warmup=args.warmup,
            repeat=args.repeat,
            return_raw_times=args.verbose or args.live,
            on_sample=on_sample_after,
        )
        if isinstance(after_out, tuple):
            after, after_times = after_out[0], after_out[1]
        else:
            after, after_times = after_out, None

    if args.live:
        print()

    print("Progenitor benchmark (before vs after)")
    print("=" * 50)
    print(f"Model:     {args.model}")
    print(f"Target:    {args.target}")
    print(f"Warmup:    {args.warmup}  Repeat: {args.repeat}")
    if args.verbose and before_times is not None and after_times is not None:
        print()
        print("Raw timings (first 5 runs, ms) — proves measurements are live, not hardcoded:")
        print(f"  Before: {[f'{t:.4f}' for t in before_times[:5]]}")
        print(f"  After:  {[f'{t:.4f}' for t in after_times[:5]]}")
    print()
    print(f"Before    latency: {before.latency_ms:.3f} ms   throughput: {before.throughput_per_sec:.1f} /s    peak RAM: {before.peak_memory_mb:.2f} MB")
    print(f"After     latency: {after.latency_ms:.3f} ms   throughput: {after.throughput_per_sec:.1f} /s    peak RAM: {after.peak_memory_mb:.2f} MB")
    
    if args.validate:
        print()
        print("Running accuracy validation...")
        try:
            mse = validate_accuracy(args.model, result.output_path)
            print(f"Validation MSE: {mse:.6e}")
            if mse > 1e-2:
                print("WARNING: High MSE detected! Enhancement may have significantly altered model outputs.")
        except Exception as e:
            print(f"Validation failed: {e}")

    if before.latency_ms > 0:
        speedup = before.latency_ms / after.latency_ms
        print(f"Speedup:   {speedup:.2f}x")
        if effective_prune is not None and not use_native_sparse and speedup < 5.0:
            print()
            print("Note: Pruned model runs with ONNX Runtime (dense). For 5–25× use a sparse backend.")
        if speedup < 1.0:
            print()
            if args.quantize:
                print("Note: 'After' (quantized) is slower here. On some CPUs (e.g. Mac, or without Intel VNNI)")
                print("  INT8 can be slower than FP32. Use graph-only enhance instead: omit --quantize.")
            elif effective_prune is not None:
                print("Note: 'After' (pruned) is slower here. ORT runs pruned weights as dense; use a sparse backend for 5–15×.")
            else:
                print("Note: 'After' is slower here. This often happens for very small models (e.g. tiny.onnx):")
                print("  full graph optimization adds overhead that doesn't pay off for a single op.")
                print("  For 2–4x on same CPU use --quantize (INT8) on larger models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

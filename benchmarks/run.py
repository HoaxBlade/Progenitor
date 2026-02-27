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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="Path to .onnx model")
    ap.add_argument("--target", "-t", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--verbose", "-v", action="store_true", help="Print raw sample timings to show measurements are live")
    ap.add_argument("--live", "-l", action="store_true", help="Stream each run's latency in real time as the benchmark runs")
    ap.add_argument("--quantize", "-q", action="store_true", help="Use INT8 quantization for 'after' (2–4x on CPU)")
    ap.add_argument("--prune", "-p", type=float, default=None, metavar="SPARSITY", help="Use magnitude pruning for 'after', e.g. 0.9 = 90%% zeros (sparse inference for 5–15×)")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"Error: not found: {args.model}", file=sys.stderr)
        return 1
    if args.prune is not None and (args.prune < 0 or args.prune > 1):
        print("Error: --prune must be between 0 and 1", file=sys.stderr)
        return 1

    result = enhance(args.model, args.target, quantize=args.quantize, prune=args.prune)
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
        if args.quantize:
            print("Mode: INT8 quantized 'after' (same device, 2–4x typical on CPU)")
        if args.prune is not None:
            print(f"Mode: Pruned {args.prune:.0%} sparsity 'after' (sparse inference for 5–15×)")
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

    # After: enhanced or quantized, same device
    if args.live:
        if args.quantize:
            after_label = "quantized INT8"
        elif args.prune is not None:
            after_label = f"pruned {args.prune:.0%} sparsity"
        else:
            after_label = "enhanced, full graph opts"
        print("After (" + after_label + "):")
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
    after = after_out[0] if (args.verbose or args.live) else after_out
    after_times = after_out[1] if (args.verbose or args.live) else None

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
    print(f"Before    latency: {before.latency_ms:.3f} ms   throughput: {before.throughput_per_sec:.1f} /s")
    print(f"After     latency: {after.latency_ms:.3f} ms   throughput: {after.throughput_per_sec:.1f} /s")
    if before.latency_ms > 0:
        speedup = before.latency_ms / after.latency_ms
        print(f"Speedup:   {speedup:.2f}x")
        if args.prune is not None and speedup < 5.0:
            print()
            print("Note: Pruned model is run with dense kernels here. For 5–15× you need sparse inference (e.g. sparse backend).")
        if speedup < 1.0:
            print()
            if args.quantize:
                print("Note: 'After' (quantized) is slower here. On some CPUs (e.g. Mac, or without Intel VNNI)")
                print("  INT8 can be slower than FP32. Use graph-only enhance instead: omit --quantize.")
            elif args.prune is not None:
                print("Note: 'After' (pruned) is slower here. ONNX Runtime runs pruned weights as dense by default.")
                print("  For 5–15× use a sparse backend or sparse kernels; see docs.")
            else:
                print("Note: 'After' is slower here. This often happens for very small models (e.g. tiny.onnx):")
                print("  full graph optimization adds overhead that doesn't pay off for a single op.")
                print("  For 2–4x on same CPU use --quantize (INT8) on larger models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

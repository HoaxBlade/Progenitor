"""CLI: progenitor enhance model.onnx --target cpu; progenitor enhance-software ./app --tune-workers."""

import argparse
import sys
from pathlib import Path

from progenitor.api import enhance


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="progenitor",
        description="Progenitor: enhance ML models (Phase 1) and software (Phase 2) to peak performance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # enhance (Phase 1: ML)
    p = subparsers.add_parser("enhance", help="Enhance an ONNX model for the given target (Phase 1).")
    p.add_argument("model", type=Path, help="Path to .onnx model")
    p.add_argument("--target", "-t", default="cpu", choices=("cpu", "cuda"), help="Hardware target (default: cpu)")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output path for enhanced model (default: <model>_enhanced.onnx)")
    p.add_argument("--quantize", action="store_true", help="Apply INT8 quantization (2–4x on CPU)")
    p.add_argument("--prune", type=float, default=None, metavar="SPARSITY", help="Magnitude pruning, e.g. 0.9 = 90%% zeros (sparse inference for 5–15×)")
    p.add_argument("--struct-prune", type=float, default=None, metavar="RATIO", help="Structured pruning: remove RATIO fraction of hidden neurons (e.g. 0.5 = 50%% removed, ~2-4×)")
    p.add_argument("--lowrank", type=float, default=None, metavar="RANK_RATIO", help="Low-rank SVD decomposition: keep RANK_RATIO of singular values (e.g. 0.25, ~2-3×)")
    p.add_argument("--max-speed", action="store_true", help="Chain all optimizations for maximum speedup (~30-50×)")
    p.set_defaults(func=_cmd_enhance)

    # enhance-software (Phase 2)
    p2 = subparsers.add_parser("enhance-software", help="Enhance a software artifact (Phase 2). Opt-in levers only.")
    p2.add_argument("artifact", type=Path, help="Path to artifact directory containing progenitor.yaml")
    p2.add_argument("--tune-workers", action="store_true", help="Enable workers lever (safe cap by CPU count)")
    p2.add_argument("--workers", type=int, default=None, metavar="N", help="Set workers to N (used with --tune-workers; otherwise auto from CPU)")
    p2.add_argument("--output-env", type=Path, default=None, help="Write env to this path (default: <artifact>/.env.progenitor)")
    p2.set_defaults(func=_cmd_enhance_software)

    args = parser.parse_args()
    args.func(args)


def _cmd_enhance(args: argparse.Namespace) -> None:
    if args.prune is not None and (args.prune < 0 or args.prune > 1):
        print("Error: --prune must be between 0 and 1 (e.g. 0.9 for 90%% sparsity)", file=sys.stderr)
        sys.exit(1)
    if args.struct_prune is not None and (args.struct_prune <= 0 or args.struct_prune >= 1):
        print("Error: --struct-prune must be between 0 and 1 (exclusive)", file=sys.stderr)
        sys.exit(1)
    if args.lowrank is not None and (args.lowrank <= 0 or args.lowrank >= 1):
        print("Error: --lowrank must be between 0 and 1 (exclusive)", file=sys.stderr)
        sys.exit(1)
    result = enhance(
        args.model,
        args.target,
        output_path=args.output,
        quantize=args.quantize,
        prune=args.prune,
        struct_prune=args.struct_prune,
        lowrank=args.lowrank,
        max_speed=args.max_speed,
    )
    if not result.compatible:
        print(result.message, file=sys.stderr)
        if "not found" in result.message.lower():
            print("", file=sys.stderr)
            print("Use the path to a real .onnx file on your machine. To try with a sample model:", file=sys.stderr)
            print("  python examples/create_tiny_onnx.py", file=sys.stderr)
            print("  progenitor enhance examples/tiny.onnx -o examples/tiny_enhanced.onnx", file=sys.stderr)
        sys.exit(1)
    print(f"Enhanced: {result.output_path}")
    print(result.message)


def _cmd_enhance_software(args: argparse.Namespace) -> None:
    from progenitor.software.enhance import enhance_software
    try:
        out = enhance_software(
            args.artifact,
            tune_workers=args.tune_workers,
            workers=args.workers,
            output_env_path=args.output_env,
        )
        if args.tune_workers:
            print(f"Enhanced: {out}")
            print("Applied: --tune-workers (WORKERS in env). Source this file before running your app.")
        else:
            print(f"Written: {out}")
            print("No levers enabled. Use --tune-workers to apply workers tuning.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

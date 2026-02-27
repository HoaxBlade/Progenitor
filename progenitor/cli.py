"""CLI: progenitor enhance model.onnx --target cpu."""

import argparse
import sys
from pathlib import Path

from progenitor.api import enhance


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="progenitor",
        description="Progenitor: enhance compatible ML models to peak performance (Phase 1: ONNX).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # enhance
    p = subparsers.add_parser("enhance", help="Enhance an ONNX model for the given target.")
    p.add_argument("model", type=Path, help="Path to .onnx model")
    p.add_argument("--target", "-t", default="cpu", choices=("cpu", "cuda"), help="Hardware target (default: cpu)")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output path for enhanced model (default: <model>_enhanced.onnx)")
    p.add_argument("--quantize", action="store_true", help="Apply INT8 quantization (2–4x on CPU)")
    p.add_argument("--prune", type=float, default=None, metavar="SPARSITY", help="Magnitude pruning, e.g. 0.9 = 90%% zeros (sparse inference for 5–15×)")
    p.add_argument("--struct-prune", type=float, default=None, metavar="RATIO", help="Structured pruning: remove RATIO fraction of hidden neurons (e.g. 0.5 = 50%% removed, ~2-4×)")
    p.add_argument("--lowrank", type=float, default=None, metavar="RANK_RATIO", help="Low-rank SVD decomposition: keep RANK_RATIO of singular values (e.g. 0.25, ~2-3×)")
    p.add_argument("--max-speed", action="store_true", help="Chain all optimizations for maximum speedup (~30-50×)")
    p.set_defaults(func=_cmd_enhance)

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

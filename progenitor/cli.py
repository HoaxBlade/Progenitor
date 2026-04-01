"""CLI: progenitor enhance model.onnx --target cpu; progenitor enhance-software ./app --tune-workers."""

import argparse
import sys
from pathlib import Path

from progenitor.api import enhance


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="progenitor",
        description="Progenitor: enhance ML models (Phase 1), software (Phase 2), and devices (Phase 3) to peak performance.",
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

    # enhance-software (Phase 2): any URL or artifact with progenitor.yaml
    p2 = subparsers.add_parser("enhance-software", help="Enhance any website (--url) or an artifact with progenitor.yaml.")
    p2.add_argument("artifact", type=Path, nargs="?", default=None, help="Path to artifact dir with progenitor.yaml (optional if --url is set)")
    p2.add_argument("--url", metavar="URL", default=None, help="Any website URL; we measure and show what to improve + commands (no YAML needed)")
    p2.add_argument("--target", default="latency", choices=("latency", "api", "payload", "caching", "all"), help="What to improve (default: latency)")
    p2.add_argument("--proxy", action="store_true", help="Start local proxy; show before/after speedup; proxy stays running until Ctrl+C")
    p2.add_argument("--api-paths", metavar="PATHS", default=None, help="Comma-separated paths to probe (e.g. /api/users,/api/items) when --target api")
    p2.add_argument("--repeat", type=int, default=20, metavar="N", help="Requests per measurement (default: 20)")
    p2.add_argument("--warmup", type=int, default=3, metavar="N", help="Warmup requests (default: 3)")
    p2.add_argument("--tune-workers", action="store_true", help="Enable workers lever (artifact mode only; safe cap by CPU count)")
    p2.add_argument("--workers", type=int, default=None, metavar="N", help="Set workers to N (used with --tune-workers)")
    p2.add_argument("--output-env", type=Path, default=None, help="Write env to this path (artifact mode; default: <artifact>/.env.progenitor)")
    p2.set_defaults(func=_cmd_enhance_software)

    # serve (Phase 2): run proxy in front of ORIGIN — customer deploys once, points domain, no code changes
    p3 = subparsers.add_parser("serve", help="Run Progenitor proxy in front of a site. Set ORIGIN to the real site URL; point your domain here. No code changes.")
    p3.add_argument("--origin", "-o", metavar="URL", default=None, help="Origin URL (e.g. https://your-site.com). Default: env ORIGIN.")
    p3.add_argument("--port", "-p", type=int, default=None, help="Port (default: 8080 or env PORT)")
    p3.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 for deployability)")
    p3.set_defaults(func=_cmd_serve)

    # enhance-device (Phase 3): measure → tune → measure on a device (mock adapter by default)
    p4 = subparsers.add_parser(
        "enhance-device",
        help="Enhance a device on your network (Phase 3). Measure baseline → apply tuning → measure after.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Linux levers:   --cpu-governor  --io-scheduler  --swappiness  --transparent-hugepages\n"
            "Windows levers: --power-plan  --disable-visual-effects  --disable-background-apps  --game-mode\n"
            "Android levers: --performance-profile  --disable-doze  --reduce-animations  --background-limits\n"
        ),
    )
    p4.add_argument("--device", "-d", metavar="ID", default=None,
                    help="Device ID (IP, hostname). Default: env PROGENITOR_DEVICE or 127.0.0.1 for dry-run.")
    p4.add_argument("--device-type", metavar="TYPE", default=None,
                    choices=("phone_android", "pc_windows", "pc_linux"),
                    help="Override device type detection (phone_android | pc_windows | pc_linux).")
    p4.add_argument("--list", action="store_true",
                    help="List discoverable devices on the LAN (uses access module if set).")
    p4.add_argument("--dry-run", action="store_true",
                    help="Run pipeline with mock adapter and simulated metrics (no real device).")
    # Linux levers
    _lg = p4.add_argument_group("Linux levers")
    _lg.add_argument("--cpu-governor", action="store_true",
                     help="Set CPU scaling governor to performance on all cores.")
    _lg.add_argument("--io-scheduler", action="store_true",
                     help="Set block-device I/O scheduler to mq-deadline.")
    _lg.add_argument("--swappiness", action="store_true",
                     help="Lower vm.swappiness to 10 (keep hot pages in RAM).")
    _lg.add_argument("--transparent-hugepages", action="store_true",
                     help="Enable transparent hugepages (reduces TLB pressure for large workloads).")
    # Windows levers
    _wg = p4.add_argument_group("Windows levers")
    _wg.add_argument("--power-plan", action="store_true",
                     help="Switch Windows power plan to High Performance.")
    _wg.add_argument("--disable-visual-effects", action="store_true",
                     help="Set visual effects to best performance (disables animations/shadows).")
    _wg.add_argument("--disable-background-apps", action="store_true",
                     help="Disable Windows background app execution via registry.")
    _wg.add_argument("--game-mode", action="store_true",
                     help="Enable Windows Game Mode (foreground resource boost).")
    # Android levers
    _ag = p4.add_argument_group("Android levers")
    _ag.add_argument("--performance-profile", action="store_true",
                     help="Set Android CPU governor to performance on all cores.")
    _ag.add_argument("--disable-doze", action="store_true",
                     help="Disable Android Doze mode (prevents aggressive background kills).")
    _ag.add_argument("--reduce-animations", action="store_true",
                     help="Set all animation scales to 0 (snappier UI).")
    _ag.add_argument("--background-limits", action="store_true",
                     help="Limit background process count to 4 (free CPU/RAM for foreground).")
    p4.set_defaults(func=_cmd_enhance_device)

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
    from progenitor.software.enhance import enhance_software, enhance_software_by_url
    if args.url:
        api_paths = [p.strip() for p in args.api_paths.split(",")] if args.api_paths else None
        try:
            enhance_software_by_url(
                args.url,
                target=args.target,
                proxy=args.proxy,
                repeat=args.repeat,
                warmup=args.warmup,
                api_paths=api_paths,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return
    # Artifact mode: requires path to dir with progenitor.yaml
    if args.artifact is None:
        print("Error: Give an artifact directory or use --url <URL> to improve any website.", file=sys.stderr)
        sys.exit(1)
    path = Path(args.artifact).resolve()
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        print("Use a real project directory, or --url <URL> for any website (no directory needed).", file=sys.stderr)
        sys.exit(1)
    if not path.is_dir():
        print(f"Error: Not a directory: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        out = enhance_software(
            path,
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


def _cmd_serve(args: argparse.Namespace) -> None:
    """Run proxy in front of ORIGIN. Customer deploys this once, sets ORIGIN, points domain — real site gets faster for everyone."""
    from progenitor.software.proxy import serve_standalone
    try:
        serve_standalone(host=args.host, port=args.port, origin=args.origin)
    except SystemExit as e:
        print(e.args[0] if e.args else "ORIGIN required.", file=sys.stderr)
        sys.exit(1)


def _cmd_enhance_device(args: argparse.Namespace) -> None:
    """Run Phase 3 pipeline: measure baseline → apply enhancements → measure after."""
    import os
    from progenitor.devices import get_default_adapter, mock_adapter, run_pipeline
    from progenitor.devices.enhance import EnhanceOptions
    from progenitor.devices.types import DeviceType

    if args.list:
        adapter = mock_adapter() if args.dry_run else get_default_adapter()
        devices = adapter.list_devices()
        if not devices:
            print("No devices discovered. Set PROGENITOR_ACCESS_MODULE to use an external discovery.")
        else:
            print("Devices (discoverable):")
            for d in devices:
                print(f"  {d}")
        return

    device_id = args.device or os.environ.get("PROGENITOR_DEVICE") or "127.0.0.1"
    if args.dry_run:
        device_id = args.device or "127.0.0.1"
    adapter = mock_adapter() if args.dry_run else get_default_adapter()

    dtype: DeviceType | None = None
    if args.device_type:
        dtype = DeviceType(args.device_type)

    opts = EnhanceOptions(
        cpu_governor=args.cpu_governor,
        io_scheduler=args.io_scheduler,
        swappiness=args.swappiness,
        transparent_hugepages=args.transparent_hugepages,
        power_plan=args.power_plan,
        disable_visual_effects=args.disable_visual_effects,
        disable_background_apps=args.disable_background_apps,
        game_mode=args.game_mode,
        performance_profile=args.performance_profile,
        disable_doze=args.disable_doze,
        reduce_animations=args.reduce_animations,
        background_limits=args.background_limits,
    )

    try:
        report = run_pipeline(device_id, adapter=adapter, opts=opts, device_type=dtype)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    b, a = report.baseline, report.after
    print(f"Device: {report.device_id}  |  Type: {report.device_type.value}")
    print()
    print("Before")
    print(f"  CPU score:         {b.cpu_score:>12.1f}")
    print(f"  I/O (MB/s):        {b.io_mb_s:>12.2f}")
    if b.latency_ms:
        print(f"  Latency (ms):      {b.latency_ms:>12.1f}")
    if b.frame_rate_fps is not None:
        print(f"  Frame rate (fps):  {b.frame_rate_fps:>12.1f}")
    if b.boot_time_s is not None:
        print(f"  Boot time (s):     {b.boot_time_s:>12.1f}")
    if b.idle_power_w is not None:
        print(f"  Idle power (W):    {b.idle_power_w:>12.1f}")
    if b.battery_drain_per_hr is not None:
        print(f"  Battery (%/hr):    {b.battery_drain_per_hr:>12.1f}")
    print()
    print("After")
    print(f"  CPU score:         {a.cpu_score:>12.1f}")
    print(f"  I/O (MB/s):        {a.io_mb_s:>12.2f}")
    if a.latency_ms:
        print(f"  Latency (ms):      {a.latency_ms:>12.1f}")
    if a.frame_rate_fps is not None:
        print(f"  Frame rate (fps):  {a.frame_rate_fps:>12.1f}")
    if a.boot_time_s is not None:
        print(f"  Boot time (s):     {a.boot_time_s:>12.1f}")
    if a.idle_power_w is not None:
        print(f"  Idle power (W):    {a.idle_power_w:>12.1f}")
    if a.battery_drain_per_hr is not None:
        print(f"  Battery (%/hr):    {a.battery_drain_per_hr:>12.1f}")
    if a.applied_changes:
        print("  Applied:")
        for change in a.applied_changes:
            print(f"    {change}")
    print()
    print(f"Speedup (CPU):       {report.speedup_cpu:.2f}x")
    print(f"Speedup (I/O):       {report.speedup_io:.2f}x")
    if b.latency_ms:
        print(f"Speedup (latency):   {report.speedup_latency:.2f}x")
    if b.frame_rate_fps:
        print(f"Speedup (fps):       {report.speedup_frame_rate:.2f}x")
    if report.improvement_boot_s:
        print(f"Boot time saved:     {report.improvement_boot_s:.1f}s")
    if report.improvement_battery_pct:
        print(f"Battery saved:       {report.improvement_battery_pct:.1f}%/hr")
    print()
    print(report.message)

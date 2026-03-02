"""Apply opt-in tuning levers to a software artifact, or enhance any website by URL."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from progenitor.software.manifest import load_manifest, LeverSpec

ENHANCED_ENV_FILENAME = ".env.progenitor"


def _safe_workers_value(spec: LeverSpec, explicit_value: int | None = None) -> int:
    """Choose a safe worker count: explicit, or capped by CPU count and spec."""
    if explicit_value is not None:
        return max(spec.min_value, min(spec.max_value, int(explicit_value)))
    try:
        cpu_count = len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        cpu_count = os.cpu_count() or 2
    return max(spec.min_value, min(spec.max_value, int(cpu_count)))


def enhance_software(
    artifact_path: str | Path,
    *,
    tune_workers: bool = False,
    workers: int | None = None,
    output_env_path: str | Path | None = None,
) -> Path:
    """
    Apply only the levers you enable. Writes an env file (e.g. .env.progenitor) that the
    run command can source. No automatic "improve everything" — each lever is opt-in.

    Returns path to the written env file.
    """
    manifest = load_manifest(artifact_path)
    out_path = Path(output_env_path) if output_env_path else manifest.artifact_path / ENHANCED_ENV_FILENAME
    out_path = out_path.resolve()

    env_lines: list[str] = []
    if tune_workers and "workers" in manifest.tune:
        spec = manifest.tune["workers"]
        value = _safe_workers_value(spec, workers)
        env_lines.append(f"WORKERS={value}")
        env_lines.append("")

    if not env_lines:
        env_lines = ["# Progenitor: no levers enabled. Use --tune-workers (or other --tune-* flags) to apply.\n"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(env_lines), encoding="utf-8")
    return out_path


def _format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024*1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def enhance_software_by_url(
    url: str,
    *,
    target: str = "latency",
    proxy: bool = False,
    repeat: int = 20,
    warmup: int = 3,
    api_paths: list[str] | None = None,
) -> None:
    """
    Measure any website, analyze for the chosen target, and print what can be improved
    plus copy-pasteable commands. All output to stdout (no report file).
    If proxy=True, start local proxy and show real before/after speedup; proxy stays running until Ctrl+C.
    """
    from progenitor.software.measure import measure, measure_api
    from progenitor.software.analyze import analyze
    from progenitor.software.proxy import run_proxy_and_measure

    url = url.strip().rstrip("/") or "/"
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    print(f"Measuring {url} ({repeat} requests, warmup {warmup})...")
    print()

    m = measure(url, warmup=warmup, repeat=repeat)
    if api_paths and target == "api":
        api_timings = measure_api(url, api_paths, warmup=2, repeat=5)
        from progenitor.software.measure import SiteMeasurement
        m = SiteMeasurement(
            url=m.url,
            ttfb_ms=m.ttfb_ms,
            total_ms=m.total_ms,
            size_bytes=m.size_bytes,
            compressed=m.compressed,
            cache_control=m.cache_control,
            etag=m.etag,
            status_code=m.status_code,
            server=m.server,
            powered_by=m.powered_by,
            p99_ms=m.p99_ms,
            raw_times_ms=m.raw_times_ms,
            api_timings=api_timings,
        )

    report = analyze(m, target)

    # Print before
    print("Target:", target)
    print()
    print("Before")
    print("  TTFB (median):   ", f"{report.before.ttfb_ms:.0f} ms", "  p99:", f"{report.before.p99_ms:.0f} ms")
    print("  Total (median):  ", f"{report.before.total_ms:.0f} ms")
    print("  Response size:   ", _format_size(report.before.size_bytes), "  Compression:", "yes" if report.before.compressed else "none")
    print("  Cache-Control:   ", report.before.cache_control or "(none)")
    print("  Server:          ", report.before.server or "(none)")
    print()

    if report.findings:
        print("These can be improved")
        for f in report.findings:
            print("  [{}]  {}".format(f.severity, f.detail))
        print()
        print("Commands that will do it")
        for i, f in enumerate(report.findings, 1):
            print(f"  {i}.", f.detail.split(".")[0] + ".")
            for line in f.fix_commands:
                print("     ", line)
            print()
    else:
        print("No issues found for this target.")
        print()

    print("Estimated speedup:", report.estimated_speedup)
    print()

    if proxy:
        print("Measuring direct (before)...")
        before, after, speedup, px = run_proxy_and_measure(url, warmup=warmup, repeat=repeat)
        print("  Total (median):", f"{before.total_ms:.0f} ms", "  p99:", f"{before.p99_ms:.0f} ms", "  Size:", _format_size(before.size_bytes))
        print()
        print("Measuring through proxy (after, repeat visits)...")
        print("  Total (median):", f"{after.total_ms:.0f} ms", "  p99:", f"{after.p99_ms:.0f} ms", "  Size:", _format_size(after.size_bytes))
        print()
        print(f"Speedup: {speedup:.1f}x (repeat visits from cache)")
        print()
        print(f"Proxy running at {px.base_url()} — use this URL instead of the original. Ctrl+C to stop.")
        print()

        def shutdown(sig=None, frame=None):
            px.stop()
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, shutdown)
            signal.signal(signal.SIGTERM, shutdown)
        except (AttributeError, ValueError):
            pass  # Windows or unsupported
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown()

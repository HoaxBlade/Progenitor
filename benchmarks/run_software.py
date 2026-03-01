#!/usr/bin/env python3
"""
Phase 2 benchmark: measure latency and throughput of a URL (e.g. your app).
You start the server yourself (before/after config); we only hit the URL and report.

Usage:
  python benchmarks/run_software.py --url http://127.0.0.1:8000/ --repeat 50
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark a URL (Phase 2 software).")
    ap.add_argument("--url", required=True, help="Base URL to request (e.g. http://127.0.0.1:8000/)")
    ap.add_argument("--repeat", type=int, default=50, help="Number of requests (default 50)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup requests (default 5)")
    args = ap.parse_args()

    try:
        import urllib.request
    except ImportError:
        try:
            import urllib2 as urllib  # type: ignore
            urllib.request = urllib
        except ImportError:
            print("Error: need urllib (standard library)", file=sys.stderr)
            return 1

    url = args.url.rstrip("/") + "/"
    times_ms: list[float] = []

    for i in range(args.warmup + args.repeat):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                r.read()
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            return 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i >= args.warmup:
            times_ms.append(elapsed_ms)

    n = len(times_ms)
    total_ms = sum(times_ms)
    avg_ms = total_ms / n if n else 0
    times_sorted = sorted(times_ms)
    p50 = times_sorted[n // 2] if n else 0
    p99 = times_sorted[int(n * 0.99)] if n >= 2 else times_sorted[-1] if n else 0
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

    print("Phase 2 software benchmark")
    print("=" * 40)
    print(f"URL:      {url}")
    print(f"Requests: {n} (warmup {args.warmup})")
    print(f"Latency:  avg {avg_ms:.2f} ms  p50 {p50:.2f} ms  p99 {p99:.2f} ms")
    print(f"Throughput: {throughput:.1f} /s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

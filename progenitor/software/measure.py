"""Real HTTP measurement for Phase 2: GET requests, TTFB, total time, size, headers."""

from __future__ import annotations

import ssl
import time
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError

# Optional certifi for SSL
def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _is_ssl_failure(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError):
        return True
    return False


@dataclass
class SiteMeasurement:
    """Measured metrics for a URL (from real GET requests)."""

    url: str
    ttfb_ms: float
    total_ms: float
    size_bytes: int
    compressed: bool
    cache_control: str
    etag: bool
    status_code: int
    server: str
    powered_by: str
    p99_ms: float
    raw_times_ms: list[float]
    api_timings: dict[str, float]


def _fetch_one(
    url: str,
    timeout: int = 15,
) -> tuple[float, float, int, dict[str, str], int]:
    """One GET request. Returns (ttfb_ms, total_ms, size_bytes, headers_lower, status_code)."""
    ctx = _ssl_context()
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "Progenitor/1.0")
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            ttfb_ms = (time.perf_counter() - t0) * 1000.0
            body = r.read()
            total_ms = (time.perf_counter() - t0) * 1000.0
            headers = {k.lower(): v for k, v in r.headers.items()}
            return ttfb_ms, total_ms, len(body), headers, r.status
    except Exception as e:
        if _is_ssl_failure(e):
            ctx = ssl._create_unverified_context()
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                ttfb_ms = (time.perf_counter() - t0) * 1000.0
                body = r.read()
                total_ms = (time.perf_counter() - t0) * 1000.0
                headers = {k.lower(): v for k, v in r.headers.items()}
                return ttfb_ms, total_ms, len(body), headers, r.status
        raise


def measure(
    url: str,
    warmup: int = 3,
    repeat: int = 20,
    timeout: int = 15,
) -> SiteMeasurement:
    """
    Measure a URL with real GET requests. Returns SiteMeasurement with
    TTFB, total time, size, compression, cache headers, p99, raw times.
    """
    url = url.rstrip("/") or url
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    # Warmup
    for _ in range(warmup):
        try:
            _fetch_one(url, timeout=timeout)
        except Exception:
            pass

    # Timed runs
    ttfb_list: list[float] = []
    total_list: list[float] = []
    size_bytes = 0
    headers_agg: dict[str, str] = {}
    status_code = 200

    for _ in range(repeat):
        ttfb_ms, total_ms, size, headers, status = _fetch_one(url, timeout=timeout)
        ttfb_list.append(ttfb_ms)
        total_list.append(total_ms)
        size_bytes = size
        headers_agg = headers
        status_code = status

    raw_times_ms = total_list
    n = len(raw_times_ms)
    total_sorted = sorted(raw_times_ms)
    ttfb_sorted = sorted(ttfb_list)
    p50_total = total_sorted[n // 2] if n else 0.0
    p50_ttfb = ttfb_sorted[n // 2] if n else 0.0
    p99_ms = total_sorted[min(n - 1, int(n * 0.99))] if n else 0.0

    ce = (headers_agg.get("content-encoding") or "").lower()
    compressed = "gzip" in ce or "br" in ce or "deflate" in ce
    cache_control = headers_agg.get("cache-control") or ""
    etag = bool(headers_agg.get("etag"))
    server = headers_agg.get("server") or ""
    powered_by = headers_agg.get("x-powered-by") or ""

    return SiteMeasurement(
        url=url,
        ttfb_ms=p50_ttfb,
        total_ms=p50_total,
        size_bytes=size_bytes,
        compressed=compressed,
        cache_control=cache_control,
        etag=etag,
        status_code=status_code,
        server=server,
        powered_by=powered_by,
        p99_ms=p99_ms,
        raw_times_ms=raw_times_ms,
        api_timings={},
    )


def measure_api(
    base_url: str,
    paths: list[str],
    warmup: int = 2,
    repeat: int = 5,
    timeout: int = 15,
) -> dict[str, float]:
    """
    Probe multiple paths (e.g. /api/users, /api/items). Returns path -> median ms.
    """
    base_url = base_url.rstrip("/")
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "https://" + base_url
    result: dict[str, float] = {}
    for path in paths:
        path = path if path.startswith("/") else "/" + path
        url = base_url + path
        times_ms: list[float] = []
        for _ in range(warmup):
            try:
                _fetch_one(url, timeout=timeout)
            except Exception:
                pass
        for _ in range(repeat):
            try:
                _, total_ms, _, _, _ = _fetch_one(url, timeout=timeout)
                times_ms.append(total_ms)
            except Exception:
                pass
        if times_ms:
            times_sorted = sorted(times_ms)
            n = len(times_sorted)
            result[path] = times_sorted[n // 2]
    return result

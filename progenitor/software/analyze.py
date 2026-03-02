"""Analyze SiteMeasurement and produce findings with copy-pasteable fix_commands."""

from __future__ import annotations

from dataclasses import dataclass

from progenitor.software.measure import SiteMeasurement


@dataclass
class Finding:
    """One improvement opportunity with concrete fix commands."""

    dimension: str
    severity: str  # "HIGH" | "MED" | "INFO"
    detail: str
    fix_commands: list[str]


@dataclass
class EnhanceReport:
    """Result of analysis: before metrics, findings (each with fix_commands), estimated speedup."""

    url: str
    target: str
    before: SiteMeasurement
    findings: list[Finding]
    estimated_speedup: str


def _stack_hint(m: SiteMeasurement) -> str:
    """Infer stack from server/x-powered-by for stack-aware commands."""
    s = (m.server + " " + m.powered_by).lower()
    if "nginx" in s:
        return "nginx"
    if "apache" in s:
        return "apache"
    if "gunicorn" in s:
        return "gunicorn"
    if "express" in s or "node" in s:
        return "express"
    if "next.js" in s:
        return "nextjs"
    return "generic"


def _format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024*1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def analyze(measurement: SiteMeasurement, target: str) -> EnhanceReport:
    """
    Analyze a SiteMeasurement for the given target (latency, api, payload, caching, all).
    Returns EnhanceReport with findings and fix_commands (copy-pasteable).
    """
    m = measurement
    findings: list[Finding] = []
    stack = _stack_hint(m)

    def add(dimension: str, severity: str, detail: str, fix_commands: list[str]) -> None:
        findings.append(Finding(dimension=dimension, severity=severity, detail=detail, fix_commands=fix_commands))

    # Compression
    if target in ("latency", "payload", "all") and not m.compressed and m.size_bytes > 1024:
        size_str = _format_size(m.size_bytes)
        est_saved = int(m.size_bytes * 0.85)
        add(
            "compression",
            "HIGH",
            f"No compression. Response is {size_str} raw; gzip would reduce to ~{_format_size(est_saved)} (~85% saving).",
            _fix_compression(stack),
        )

    # Cache-Control
    if target in ("latency", "caching", "all"):
        cc = (m.cache_control or "").lower()
        if "no-cache" in cc or "no-store" in cc or not m.cache_control:
            size_str = _format_size(m.size_bytes)
            add(
                "caching",
                "HIGH",
                f"Cache-Control: {m.cache_control or 'missing'} — every visit re-fetches {size_str}.",
                _fix_cache_control(stack),
            )

    # ETag
    if target in ("latency", "caching", "all") and not m.etag:
        add(
            "caching",
            "MED",
            "No ETag header — conditional requests (304 Not Modified) not possible.",
            _fix_etag(stack),
        )

    # Latency / TTFB (informational if high)
    if target in ("latency", "all") and m.ttfb_ms > 500:
        add(
            "latency",
            "MED",
            f"TTFB (median) is {m.ttfb_ms:.0f} ms. Consider server-side caching or a CDN.",
            ["Add a reverse proxy (e.g. Nginx) with proxy_cache.", "Or use a CDN in front of your origin."],
        )

    # API timings (if we have per-path data)
    if target == "api" and m.api_timings:
        for path, median_ms in sorted(m.api_timings.items(), key=lambda x: -x[1]):
            if median_ms > 500:
                add(
                    "api",
                    "MED",
                    f"Path {path}: median {median_ms:.0f} ms.",
                    [f"Optimize backend for {path} (indexes, caching, or smaller payload)."],
                )

    # Estimated speedup
    if findings:
        if any(f.dimension == "compression" for f in findings) and any(f.dimension == "caching" for f in findings):
            estimated_speedup = "3–6x on repeat visits (compression + caching)."
        elif any(f.dimension == "compression" for f in findings):
            estimated_speedup = "2–4x (compression reduces payload)."
        elif any(f.dimension == "caching" for f in findings):
            estimated_speedup = "2–4x on repeat visits (caching)."
        else:
            estimated_speedup = "Improvement depends on applying the commands above."
    else:
        estimated_speedup = "No major issues detected for this target."

    return EnhanceReport(
        url=m.url,
        target=target,
        before=m,
        findings=findings,
        estimated_speedup=estimated_speedup,
    )


def _fix_compression(stack: str) -> list[str]:
    if stack == "nginx":
        return [
            "Add to your nginx server block:",
            "  gzip on;",
            "  gzip_types text/html application/json application/javascript;",
            "Reload: sudo nginx -s reload",
        ]
    if stack == "apache":
        return [
            "Enable mod_deflate. In .htaccess or config:",
            "  SetOutputFilter DEFLATE",
            "  SetEnvIfNoCase Request_URI \\.(?:gif|jpe?g|png)$ no-gzip",
        ]
    if stack == "express":
        return [
            "npm install compression",
            "In your app: const compression = require('compression'); app.use(compression());",
        ]
    if stack == "gunicorn":
        return [
            "Gunicorn does not compress; put Nginx in front with gzip on; gzip_types text/html application/json;",
        ]
    return [
        "Enable gzip or brotli on your server.",
        "Nginx: gzip on; gzip_types text/html application/json;",
        "Apache: mod_deflate. Express: app.use(require('compression')());",
    ]


def _fix_cache_control(stack: str) -> list[str]:
    if stack == "nginx":
        return [
            "For cacheable pages add in location block: add_header Cache-Control \"max-age=3600\";",
            "For static assets: add_header Cache-Control \"max-age=31536000, public\";",
        ]
    if stack == "express":
        return [
            "Set header in route or middleware: res.set('Cache-Control', 'max-age=3600');",
            "For static: app.use(express.static('public', { maxAge: '1y' }));",
        ]
    return [
        "Add Cache-Control header: max-age=3600 for pages, max-age=31536000 for static assets.",
    ]


def _fix_etag(stack: str) -> list[str]:
    if stack == "nginx":
        return ["In location block: etag on;"]
    if stack in ("express", "gunicorn"):
        return ["Express/Flask/Django enable ETag by default; ensure you are not overriding with Cache-Control: no-store."]
    return ["Enable ETag in your server or framework so clients can send If-None-Match for 304 responses."]

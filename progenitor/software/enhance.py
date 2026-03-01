"""Apply opt-in tuning levers to a software artifact, or recommend improvements for any URL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from progenitor.software.manifest import load_manifest, SoftwareManifest, LeverSpec

ENHANCED_ENV_FILENAME = ".env.progenitor"
RECOMMENDATIONS_FILENAME = "progenitor-recommendations.txt"


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
    # Only set what was requested
    if tune_workers and "workers" in manifest.tune:
        spec = manifest.tune["workers"]
        value = _safe_workers_value(spec, workers)
        env_lines.append(f"WORKERS={value}")
        env_lines.append("")

    if not env_lines:
        # Write a comment so the file exists and user sees nothing was auto-applied
        env_lines = ["# Progenitor: no levers enabled. Use --tune-workers (or other --tune-* flags) to apply.\n"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(env_lines), encoding="utf-8")
    return out_path


def _fetch_headers(url: str, timeout: int = 15) -> dict[str, str]:
    """HEAD request; return lowercase header dict. Handles SSL on macOS."""
    import ssl
    import urllib.request
    from urllib.error import URLError

    def _open(ctx: ssl.SSLContext | None) -> dict[str, str]:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)) if ctx else urllib.request.build_opener()
        req = urllib.request.Request(url, method="HEAD")
        with opener.open(req, timeout=timeout) as r:
            return {k.lower(): v for k, v in r.headers.items()}

    def _is_ssl_failure(exc: BaseException) -> bool:
        if isinstance(exc, ssl.SSLError):
            return True
        if isinstance(exc, URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return True
        return False

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        return _open(ctx)
    except Exception as e:
        if _is_ssl_failure(e):
            ctx = ssl._create_unverified_context()
            return _open(ctx)
        raise


def _detect_stack(headers: dict[str, str]) -> list[str]:
    """Infer stack from response headers."""
    detected: list[str] = []
    powered = (headers.get("x-powered-by") or "").lower()
    server = (headers.get("server") or "").lower()
    if "next.js" in powered or "next" in powered:
        detected.append("Next.js")
    if "vercel" in server:
        detected.append("Vercel")
    if "express" in powered:
        detected.append("Express")
    if "node" in powered or "node" in server:
        detected.append("Node")
    if "nginx" in server:
        detected.append("Nginx")
    if "gunicorn" in server or "gunicorn" in powered:
        detected.append("Gunicorn")
    if "apache" in server:
        detected.append("Apache")
    if not detected:
        detected.append("Unknown (generic HTTP)")
    return detected


def _recommendations_for_stack(stack: list[str]) -> list[str]:
    """Concrete recommendations by detected stack."""
    recs: list[str] = []
    s = " ".join(stack).lower()
    if "next.js" in s or "vercel" in s:
        recs.append("Next.js: Use ISR (revalidate) or static generation for pages that don’t need per-request data.")
        recs.append("Next.js: Enable image optimization (next/image) and consider smaller image sizes.")
        recs.append("Vercel: Enable Edge where possible; use caching headers for static assets.")
        recs.append("Next.js: Reduce client-side JS with dynamic imports and avoid large dependencies on first load.")
    if "express" in s or "node" in s:
        recs.append("Node: Enable gzip/brotli compression (e.g. compression middleware).")
        recs.append("Node: Set Cache-Control headers for static and cacheable responses.")
        recs.append("Node: Consider clustering (cluster module) to use all CPU cores.")
    if "nginx" in s:
        recs.append("Nginx: Enable gzip and static asset caching; tune worker_connections and keepalive.")
    if "gunicorn" in s:
        recs.append("Gunicorn: Tune workers (e.g. 2–4 × CPU cores) and use a reverse proxy (Nginx) for static files.")
    recs.append("General: Prefer CDN/caching for static assets and set long cache headers where safe.")
    return recs


def enhance_software_by_url(
    url: str,
    output_path: str | Path | None = None,
) -> Path:
    """
    Improve any website: fetch URL, detect stack from headers, write actionable recommendations.
    No progenitor.yaml or artifact path required.
    """
    url = url.rstrip("/") or "/"
    headers = _fetch_headers(url)
    stack = _detect_stack(headers)
    recs = _recommendations_for_stack(stack)
    out = Path(output_path).resolve() if output_path else Path.cwd() / RECOMMENDATIONS_FILENAME
    lines = [
        f"# Progenitor recommendations for {url}",
        "",
        "Detected stack: " + ", ".join(stack),
        "",
        "Recommendations:",
        "",
    ]
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

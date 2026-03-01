"""Apply opt-in tuning levers to a software artifact. Writes enhanced env/config only for levers you enable."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from progenitor.software.manifest import load_manifest, SoftwareManifest, LeverSpec

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

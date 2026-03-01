"""Load and validate Phase 2 software artifact manifest (progenitor.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "progenitor.yaml"


@dataclass
class LeverSpec:
    """One tunable lever (e.g. workers) with safe min/max/default."""

    name: str
    min_value: int | float
    max_value: int | float
    default: int | float


@dataclass
class SoftwareManifest:
    """Parsed progenitor.yaml for a software artifact."""

    artifact_path: Path
    type: str  # e.g. python_http
    run_cmd: str
    tune: dict[str, LeverSpec]

    @classmethod
    def from_dict(cls, artifact_path: Path, data: dict[str, Any]) -> "SoftwareManifest":
        tune_raw = data.get("tune") or {}
        tune: dict[str, LeverSpec] = {}
        for name, spec in tune_raw.items():
            if isinstance(spec, dict):
                tune[name] = LeverSpec(
                    name=name,
                    min_value=spec.get("min", 1),
                    max_value=spec.get("max", 64),
                    default=spec.get("default", 2),
                )
            else:
                tune[name] = LeverSpec(name=name, min_value=1, max_value=64, default=2)
        return cls(
            artifact_path=artifact_path,
            type=data.get("type", "python_http"),
            run_cmd=data.get("run_cmd", ""),
            tune=tune,
        )


def load_manifest(artifact_path: str | Path) -> SoftwareManifest:
    """Load progenitor.yaml from the artifact directory. Raises if not found or invalid."""
    path = Path(artifact_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact path is not a directory: {path}")
    manifest_file = path / MANIFEST_FILENAME
    if not manifest_file.exists():
        raise FileNotFoundError(f"No {MANIFEST_FILENAME} in {path}")
    try:
        import yaml
        with open(manifest_file) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Invalid {MANIFEST_FILENAME}: {e}") from e
    return SoftwareManifest.from_dict(path, data)

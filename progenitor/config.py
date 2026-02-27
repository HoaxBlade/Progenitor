"""Target and compatibility configuration for Phase 1."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TargetId = Literal["cpu", "cuda"]


@dataclass
class Target:
    """Hardware target for inference."""

    id: TargetId
    execution_providers: tuple[str, ...]

    @classmethod
    def from_id(cls, target_id: str) -> "Target":
        id_ = target_id.lower().strip()
        if id_ == "cpu":
            return cls(id="cpu", execution_providers=("CPUExecutionProvider",))
        if id_ == "cuda":
            return cls(
                id="cuda",
                execution_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            )
        raise ValueError(f"Unsupported target: {target_id}. Use 'cpu' or 'cuda'.")


@dataclass
class EnhanceOptions:
    """Options for the enhance pipeline."""

    target: Target
    output_path: Path | None = None
    quantize: bool = False
    static_quantize: bool = False
    prune: float | None = None  # e.g. 0.9 = 90% sparsity (zeros)
    struct_prune: float | None = None  # e.g. 0.5 = remove 50% of hidden neurons
    conv_prune: float | None = None  # e.g. 0.5 = remove 50% of Conv bottleneck channels
    lowrank: float | None = None  # e.g. 0.25 = keep top 25% singular values
    max_speed: bool = False  # chain all passes for maximum speedup
    graph_optimization_level: int = 99  # ORT_ALL

"""Optimization passes for Progenitor. Phase 1: ML (ONNX)."""

from progenitor.optimizations.passes import apply_shape_inference
from progenitor.optimizations.prune import apply_pruning
from progenitor.optimizations.structured_prune import apply_structured_pruning
from progenitor.optimizations.lowrank import apply_lowrank_decomposition
from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
from progenitor.optimizations.block_removal import apply_block_removal

__all__ = [
    "apply_shape_inference",
    "apply_pruning",
    "apply_structured_pruning",
    "apply_lowrank_decomposition",
    "apply_conv_structured_pruning",
    "apply_block_removal",
]

"""Optimization passes for Progenitor. Phase 1: ML (ONNX)."""

from progenitor.optimizations.passes import apply_shape_inference
from progenitor.optimizations.prune import (
    apply_pruning,
    apply_importance_pruning,
    apply_block_pruning,
    tune_per_layer_sparsity,
)
from progenitor.optimizations.structured_prune import apply_structured_pruning
from progenitor.optimizations.lowrank import apply_lowrank_decomposition
from progenitor.optimizations.conv_prune import apply_conv_structured_pruning
from progenitor.optimizations.block_removal import apply_block_removal
from progenitor.optimizations.transformer_prune import apply_transformer_structured_pruning
from progenitor.optimizations.calibrate import apply_output_calibration

__all__ = [
    "apply_shape_inference",
    "apply_pruning",
    "apply_importance_pruning",
    "apply_block_pruning",
    "tune_per_layer_sparsity",
    "apply_structured_pruning",
    "apply_lowrank_decomposition",
    "apply_conv_structured_pruning",
    "apply_block_removal",
    "apply_transformer_structured_pruning",
    "apply_output_calibration",
]

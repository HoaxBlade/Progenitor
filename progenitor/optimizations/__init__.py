"""Optimization passes for Progenitor. Phase 1: ML (ONNX)."""

from progenitor.optimizations.passes import apply_shape_inference
from progenitor.optimizations.prune import apply_pruning

__all__ = ["apply_shape_inference", "apply_pruning"]

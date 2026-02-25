"""Optimization passes for Progenitor. Phase 1: ML (ONNX)."""

from progenitor.optimizations.passes import apply_shape_inference

__all__ = ["apply_shape_inference"]

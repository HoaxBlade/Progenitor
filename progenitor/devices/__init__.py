"""
Phase 3: Enhance devices to peak performance.

Enhancement pipeline: measure baseline → apply tuning → measure after.
Access to the device is via an external access module; this package provides
the pipeline and operator interface.
"""

from progenitor.devices.adapter import (
    AccessAdapter,
    DeviceSession,
    PayloadResult,
    get_default_adapter,
    mock_adapter,
)
from progenitor.devices.types import (
    DeviceBaseline,
    DeviceAfter,
    DeviceReport,
    DeviceType,
)
from progenitor.devices.measure import measure_baseline, measure_after
from progenitor.devices.enhance import apply_enhancements
from progenitor.devices.pipeline import run_pipeline

__all__ = [
    "AccessAdapter",
    "DeviceSession",
    "PayloadResult",
    "get_default_adapter",
    "mock_adapter",
    "DeviceBaseline",
    "DeviceAfter",
    "DeviceReport",
    "DeviceType",
    "measure_baseline",
    "measure_after",
    "apply_enhancements",
    "run_pipeline",
]

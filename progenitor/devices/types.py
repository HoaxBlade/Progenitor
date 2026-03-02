"""Data types for Phase 3 device enhancement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeviceType(str, Enum):
    """Supported device type (first slice: phones, PCs)."""

    PHONE_ANDROID = "phone_android"
    PC_WINDOWS = "pc_windows"
    PC_LINUX = "pc_linux"
    UNKNOWN = "unknown"


@dataclass
class DeviceBaseline:
    """Baseline metrics collected from a device before enhancement."""

    device_id: str
    device_type: DeviceType
    # Generic metrics (device-type specific keys can be added)
    cpu_score: float = 0.0  # Relative score or ops/sec
    io_mb_s: float = 0.0  # Disk/IO throughput MB/s
    latency_ms: float = 0.0  # Representative latency if applicable
    battery_drain_per_hr: float | None = None  # % per hour (phones) or None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceAfter:
    """Metrics collected after enhancement."""

    device_id: str
    device_type: DeviceType
    cpu_score: float = 0.0
    io_mb_s: float = 0.0
    latency_ms: float = 0.0
    battery_drain_per_hr: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    applied_changes: list[str] = field(default_factory=list)


@dataclass
class DeviceReport:
    """Before/after report for one device."""

    device_id: str
    device_type: DeviceType
    baseline: DeviceBaseline
    after: DeviceAfter
    speedup_cpu: float = 1.0
    speedup_io: float = 1.0
    speedup_latency: float = 1.0  # >1 means faster (lower latency → higher ratio)
    message: str = ""

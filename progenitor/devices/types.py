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
    PC_MACOS = "pc_macos"
    UNKNOWN = "unknown"


@dataclass
class DeviceBaseline:
    """Baseline metrics collected from a device before enhancement."""

    device_id: str
    device_type: DeviceType

    # Generic (all device types)
    cpu_score: float = 0.0      # Relative score or ops/sec
    io_mb_s: float = 0.0        # Disk/IO throughput MB/s

    # Latency-style metric (varies per device)
    latency_ms: float = 0.0

    # Phone-specific
    battery_drain_per_hr: float | None = None   # % per hour (None if not applicable)
    frame_rate_fps: float | None = None          # Frames per second (UI benchmark)

    # PC-specific
    boot_time_s: float | None = None            # Boot-to-ready in seconds
    idle_power_w: float | None = None           # Idle power consumption in Watts

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
    frame_rate_fps: float | None = None

    boot_time_s: float | None = None
    idle_power_w: float | None = None

    raw: dict[str, Any] = field(default_factory=dict)
    applied_changes: list[str] = field(default_factory=list)


@dataclass
class DeviceReport:
    """Before/after report for one device."""

    device_id: str
    device_type: DeviceType
    baseline: DeviceBaseline
    after: DeviceAfter

    # Speedup ratios (>1.0 means improvement)
    speedup_cpu: float = 1.0
    speedup_io: float = 1.0
    speedup_latency: float = 1.0        # baseline.latency / after.latency
    speedup_frame_rate: float = 1.0     # after.fps / baseline.fps (phones)
    improvement_boot_s: float = 0.0     # baseline.boot_s - after.boot_s (PCs)
    improvement_battery_pct: float = 0.0  # baseline.drain - after.drain (phones, lower=better)

    message: str = ""

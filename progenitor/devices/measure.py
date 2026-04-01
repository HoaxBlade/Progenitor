"""Device baseline and after measurement. Works with a real session or local mock."""

from __future__ import annotations

import platform
import time
from typing import Any

from progenitor.devices.types import DeviceBaseline, DeviceAfter, DeviceType
from progenitor.devices.adapter import DeviceSession, MockDeviceSession


def _infer_device_type(device_id: str) -> DeviceType:
    """Infer device type from device ID hint or the local platform."""
    d = device_id.lower()
    if "android" in d or "phone" in d:
        return DeviceType.PHONE_ANDROID
    sys = platform.system().lower()
    if "windows" in sys:
        return DeviceType.PC_WINDOWS
    if "linux" in sys:
        return DeviceType.PC_LINUX
    # macOS in dev counts as "unknown"; real devices would provide explicit type
    return DeviceType.UNKNOWN


# ---------------------------------------------------------------------------
# Local benchmark helpers (used in mock/dry-run mode)
# ---------------------------------------------------------------------------

def _local_cpu_score() -> float:
    """Tight loop to produce a reproducible relative CPU score (ops/sec)."""
    n = 500_000
    start = time.perf_counter()
    x = 0
    for i in range(n):
        x += i * (i + 1)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else 0.0


def _local_io_score() -> float:
    """Write 1 MB to a temp file and measure MB/s as an I/O proxy."""
    try:
        import tempfile
        import os
        size_mb = 1
        data = os.urandom(size_mb * 1024 * 1024)
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
        return size_mb / elapsed if elapsed > 0 else 0.0
    except Exception:
        return 0.0


def _mock_metrics_for(device_type: DeviceType) -> dict[str, Any]:
    """
    Return device-type-specific mock baseline metrics. Used in dry-run when there is
    no real device to measure. Values are typical for each device class.
    """
    cpu = _local_cpu_score()
    io = _local_io_score()

    if device_type == DeviceType.PHONE_ANDROID:
        return {
            "cpu_score": cpu * 0.35,    # phones are ~35% of a desktop core
            "io_mb_s": io * 0.15,        # eMMC/UFS is slower than SSD
            "latency_ms": 18.0,          # typical touch-to-response latency
            "battery_drain_per_hr": 12.0,  # % per hour under moderate load
            "frame_rate_fps": 48.0,       # typical 60 Hz panel, suboptimal governor
        }
    if device_type == DeviceType.PC_WINDOWS:
        return {
            "cpu_score": cpu,
            "io_mb_s": io,
            "latency_ms": 5.0,
            "boot_time_s": 28.0,          # typical Windows 10/11 HDD-ish cold boot
            "idle_power_w": 18.0,         # typical idle W (balanced plan)
        }
    if device_type == DeviceType.PC_LINUX:
        return {
            "cpu_score": cpu,
            "io_mb_s": io,
            "latency_ms": 2.5,
            "boot_time_s": 14.0,          # typical Linux systemd boot
            "idle_power_w": 12.0,
        }
    # UNKNOWN: plain local measurements
    return {
        "cpu_score": cpu,
        "io_mb_s": io,
        "latency_ms": 0.0,
    }


def _mock_metrics_after(device_type: DeviceType, baseline: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate post-enhancement metrics. Each device type gets realistic improvements
    based on the levers that would apply (we don't re-run benchmarks in mock mode).
    """
    after = dict(baseline)
    if device_type == DeviceType.PHONE_ANDROID:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.25   # governor unlock
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.10
        after["latency_ms"] = baseline.get("latency_ms", 18.0) * 0.65  # animations off
        after["battery_drain_per_hr"] = baseline.get("battery_drain_per_hr", 12.0) * 0.90  # doze off is a trade-off; small savings from fewer background kills
        after["frame_rate_fps"] = min(60.0, baseline.get("frame_rate_fps", 48.0) * 1.25)
    elif device_type == DeviceType.PC_WINDOWS:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.18
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.08
        after["latency_ms"] = baseline.get("latency_ms", 5.0) * 0.75
        after["boot_time_s"] = baseline.get("boot_time_s", 28.0) * 0.85
        after["idle_power_w"] = baseline.get("idle_power_w", 18.0) * 1.30  # high perf draws more at idle
    elif device_type == DeviceType.PC_LINUX:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.20
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.30   # mq-deadline + hugepages
        after["latency_ms"] = baseline.get("latency_ms", 2.5) * 0.70
        after["boot_time_s"] = baseline.get("boot_time_s", 14.0) * 0.92
        after["idle_power_w"] = baseline.get("idle_power_w", 12.0) * 1.15
    else:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.10
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.10
    return after


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_baseline(
    device_id: str,
    session: DeviceSession | None = None,
    device_type: DeviceType | None = None,
) -> DeviceBaseline:
    """
    Measure baseline metrics. Real session → run payload and parse results.
    MockDeviceSession / no session → use local mock benchmarks.
    """
    dtype = device_type or _infer_device_type(device_id)

    if session is not None and not isinstance(session, MockDeviceSession):
        result = session.run_payload("measure_baseline")
        if result.success and result.metrics:
            m = result.metrics
            return DeviceBaseline(
                device_id=device_id,
                device_type=dtype,
                cpu_score=m.get("cpu_score", 0.0),
                io_mb_s=m.get("io_mb_s", 0.0),
                latency_ms=m.get("latency_ms", 0.0),
                battery_drain_per_hr=m.get("battery_drain_per_hr"),
                frame_rate_fps=m.get("frame_rate_fps"),
                boot_time_s=m.get("boot_time_s"),
                idle_power_w=m.get("idle_power_w"),
                raw=m,
            )

    m = _mock_metrics_for(dtype)
    return DeviceBaseline(
        device_id=device_id,
        device_type=dtype,
        cpu_score=m.get("cpu_score", 0.0),
        io_mb_s=m.get("io_mb_s", 0.0),
        latency_ms=m.get("latency_ms", 0.0),
        battery_drain_per_hr=m.get("battery_drain_per_hr"),
        frame_rate_fps=m.get("frame_rate_fps"),
        boot_time_s=m.get("boot_time_s"),
        idle_power_w=m.get("idle_power_w"),
        raw={**m, "mock": True, "platform": platform.system()},
    )


def measure_after(
    device_id: str,
    session: DeviceSession | None = None,
    device_type: DeviceType | None = None,
    applied_changes: list[str] | None = None,
    baseline_raw: dict[str, Any] | None = None,
) -> DeviceAfter:
    """
    Measure post-enhancement metrics. Real session → run payload.
    Mock → derive from baseline_raw using device-type-specific improvement model.
    """
    dtype = device_type or _infer_device_type(device_id)

    if session is not None and not isinstance(session, MockDeviceSession):
        result = session.run_payload("measure_after")
        if result.success and result.metrics:
            m = result.metrics
            return DeviceAfter(
                device_id=device_id,
                device_type=dtype,
                cpu_score=m.get("cpu_score", 0.0),
                io_mb_s=m.get("io_mb_s", 0.0),
                latency_ms=m.get("latency_ms", 0.0),
                battery_drain_per_hr=m.get("battery_drain_per_hr"),
                frame_rate_fps=m.get("frame_rate_fps"),
                boot_time_s=m.get("boot_time_s"),
                idle_power_w=m.get("idle_power_w"),
                raw=m,
                applied_changes=applied_changes or [],
            )

    base = baseline_raw or _mock_metrics_for(dtype)
    m = _mock_metrics_after(dtype, base)
    return DeviceAfter(
        device_id=device_id,
        device_type=dtype,
        cpu_score=m.get("cpu_score", 0.0),
        io_mb_s=m.get("io_mb_s", 0.0),
        latency_ms=m.get("latency_ms", 0.0),
        battery_drain_per_hr=m.get("battery_drain_per_hr"),
        frame_rate_fps=m.get("frame_rate_fps"),
        boot_time_s=m.get("boot_time_s"),
        idle_power_w=m.get("idle_power_w"),
        raw={**m, "mock": True},
        applied_changes=applied_changes or [],
    )

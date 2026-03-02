"""Device baseline and after measurement. Works with session or local mock."""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

from progenitor.devices.types import DeviceBaseline, DeviceAfter, DeviceType
from progenitor.devices.adapter import DeviceSession, MockDeviceSession


def _infer_device_type(device_id: str) -> DeviceType:
    """Infer device type from ID or platform when running locally."""
    d = device_id.lower()
    if "android" in d or "phone" in d:
        return DeviceType.PHONE_ANDROID
    if "windows" in platform.system().lower():
        return DeviceType.PC_WINDOWS
    if "linux" in platform.system().lower():
        return DeviceType.PC_LINUX
    return DeviceType.UNKNOWN


def _local_cpu_score() -> float:
    """Rough local CPU score (ops per second). For mock/baseline comparison."""
    n = 500_000
    start = time.perf_counter()
    x = 0
    for i in range(n):
        x += i * (i + 1)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return n / elapsed


def _local_io_score() -> float:
    """Rough local 'IO' score (memory throughput MB/s). Mock only."""
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
        if elapsed > 0:
            return size_mb / elapsed
    except Exception:
        pass
    return 0.0


def measure_baseline(
    device_id: str,
    session: DeviceSession | None = None,
    device_type: DeviceType | None = None,
) -> DeviceBaseline:
    """
    Measure baseline metrics for the device. If session is provided and can run
    a real payload, use that; otherwise run a local mock (e.g. for --dry-run).
    """
    if session is not None and not isinstance(session, MockDeviceSession):
        result = session.run_payload("measure_baseline")
        if result.success and result.metrics:
            return DeviceBaseline(
                device_id=device_id,
                device_type=device_type or _infer_device_type(device_id),
                cpu_score=result.metrics.get("cpu_score", 0.0),
                io_mb_s=result.metrics.get("io_mb_s", 0.0),
                latency_ms=result.metrics.get("latency_ms", 0.0),
                battery_drain_per_hr=result.metrics.get("battery_drain_per_hr"),
                raw=result.metrics,
            )
    # Local mock: measure this machine as a stand-in
    return DeviceBaseline(
        device_id=device_id,
        device_type=device_type or _infer_device_type(device_id),
        cpu_score=_local_cpu_score(),
        io_mb_s=_local_io_score(),
        latency_ms=0.0,
        battery_drain_per_hr=None,
        raw={"mock": True, "platform": platform.system()},
    )


def measure_after(
    device_id: str,
    session: DeviceSession | None = None,
    device_type: DeviceType | None = None,
    applied_changes: list[str] | None = None,
) -> DeviceAfter:
    """
    Measure metrics after enhancement. Same as baseline but records applied_changes.
    """
    if session is not None and not isinstance(session, MockDeviceSession):
        result = session.run_payload("measure_after")
        if result.success and result.metrics:
            return DeviceAfter(
                device_id=device_id,
                device_type=device_type or _infer_device_type(device_id),
                cpu_score=result.metrics.get("cpu_score", 0.0),
                io_mb_s=result.metrics.get("io_mb_s", 0.0),
                latency_ms=result.metrics.get("latency_ms", 0.0),
                battery_drain_per_hr=result.metrics.get("battery_drain_per_hr"),
                raw=result.metrics,
                applied_changes=applied_changes or [],
            )
    # Local mock: run same local "benchmark" (may be slightly different)
    return DeviceAfter(
        device_id=device_id,
        device_type=device_type or _infer_device_type(device_id),
        cpu_score=_local_cpu_score(),
        io_mb_s=_local_io_score(),
        latency_ms=0.0,
        battery_drain_per_hr=None,
        raw={"mock": True, "platform": platform.system()},
        applied_changes=applied_changes or [],
    )

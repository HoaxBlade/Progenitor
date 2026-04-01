"""
Device baseline and after measurement.

Mock path  (MockDeviceSession / no session): local benchmarks + device-type-specific
  simulated values. Used for dry-run and tests.

Real path (SSHSession / ADBSession): runs actual benchmark commands on the device
  and parses the output into DeviceBaseline / DeviceAfter.
"""

from __future__ import annotations

import platform
import re
import time
from typing import Any

from progenitor.devices.types import DeviceBaseline, DeviceAfter, DeviceType
from progenitor.devices.adapter import DeviceSession, MockDeviceSession


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_float(s: str) -> float | None:
    """Extract the first float from a string (handles units, whitespace, etc.)."""
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def _infer_device_type(device_id: str) -> DeviceType:
    d = device_id.lower()
    if "android" in d or "phone" in d:
        return DeviceType.PHONE_ANDROID
    sys = platform.system().lower()
    if "windows" in sys:
        return DeviceType.PC_WINDOWS
    if "linux" in sys:
        return DeviceType.PC_LINUX
    return DeviceType.UNKNOWN


# ---------------------------------------------------------------------------
# Local benchmark helpers (mock / dry-run only)
# ---------------------------------------------------------------------------

def _local_cpu_score() -> float:
    n = 500_000
    start = time.perf_counter()
    x = 0
    for i in range(n):
        x += i * (i + 1)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else 0.0


def _local_io_score() -> float:
    try:
        import tempfile, os
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
    cpu = _local_cpu_score()
    io = _local_io_score()
    if device_type == DeviceType.PHONE_ANDROID:
        return {
            "cpu_score": cpu * 0.35,
            "io_mb_s": io * 0.15,
            "latency_ms": 18.0,
            "battery_drain_per_hr": 12.0,
            "frame_rate_fps": 48.0,
        }
    if device_type == DeviceType.PC_WINDOWS:
        return {
            "cpu_score": cpu,
            "io_mb_s": io,
            "latency_ms": 5.0,
            "boot_time_s": 28.0,
            "idle_power_w": 18.0,
        }
    if device_type == DeviceType.PC_LINUX:
        return {
            "cpu_score": cpu,
            "io_mb_s": io,
            "latency_ms": 2.5,
            "boot_time_s": 14.0,
            "idle_power_w": 12.0,
        }
    return {"cpu_score": cpu, "io_mb_s": io, "latency_ms": 0.0}


def _mock_metrics_after(device_type: DeviceType, baseline: dict[str, Any]) -> dict[str, Any]:
    after = dict(baseline)
    if device_type == DeviceType.PHONE_ANDROID:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.25
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.10
        after["latency_ms"] = baseline.get("latency_ms", 18.0) * 0.65
        after["battery_drain_per_hr"] = baseline.get("battery_drain_per_hr", 12.0) * 0.90
        after["frame_rate_fps"] = min(60.0, baseline.get("frame_rate_fps", 48.0) * 1.25)
    elif device_type == DeviceType.PC_WINDOWS:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.18
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.08
        after["latency_ms"] = baseline.get("latency_ms", 5.0) * 0.75
        after["boot_time_s"] = baseline.get("boot_time_s", 28.0) * 0.85
        after["idle_power_w"] = baseline.get("idle_power_w", 18.0) * 1.30
    elif device_type == DeviceType.PC_LINUX:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.20
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.30
        after["latency_ms"] = baseline.get("latency_ms", 2.5) * 0.70
        after["boot_time_s"] = baseline.get("boot_time_s", 14.0) * 0.92
        after["idle_power_w"] = baseline.get("idle_power_w", 12.0) * 1.15
    else:
        after["cpu_score"] = baseline.get("cpu_score", 0) * 1.10
        after["io_mb_s"] = baseline.get("io_mb_s", 0) * 1.10
    return after


# ---------------------------------------------------------------------------
# Real measurement — Linux (SSH)
# ---------------------------------------------------------------------------

# Inline Python so there are no script files to transfer
_LINUX_CPU_CMD = (
    "python3 -c 'import time; n=500000; t=time.perf_counter(); "
    "[i*(i+1) for i in range(n)]; print(n/(time.perf_counter()-t))'"
)
_LINUX_IO_CMD = (
    "python3 -c 'import os,tempfile,time; d=os.urandom(1048576); "
    "f=tempfile.NamedTemporaryFile(delete=True); t=time.perf_counter(); "
    "f.write(d); f.flush(); os.fsync(f.fileno()); print(1/(time.perf_counter()-t))'"
)
_LINUX_LATENCY_CMD = (
    "python3 -c 'import time; t=time.perf_counter(); "
    "[i for i in range(10000)]; print((time.perf_counter()-t)*1000)'"
)
# systemd-analyze gives: "Startup finished in 1.2s (kernel) + 4.5s (userspace) = 5.7s."
_LINUX_BOOT_CMD = (
    "systemd-analyze 2>/dev/null | grep -oP '[\\d.]+(?=s\\.?\\s*$)' | tail -1 || echo 0"
)
_LINUX_IDLE_POWER_CMD = (
    # Read from RAPL if available (µW → W); else skip
    "cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null | head -1 || echo ''"
)


def _measure_linux_real(session: DeviceSession, device_id: str) -> dict[str, Any]:
    m: dict[str, Any] = {}

    r = session.run_payload(_LINUX_CPU_CMD)
    m["cpu_score"] = _parse_float(r.stdout) or 0.0

    r = session.run_payload(_LINUX_IO_CMD)
    m["io_mb_s"] = _parse_float(r.stdout) or 0.0

    r = session.run_payload(_LINUX_LATENCY_CMD)
    m["latency_ms"] = _parse_float(r.stdout) or 0.0

    r = session.run_payload(_LINUX_BOOT_CMD)
    m["boot_time_s"] = _parse_float(r.stdout)  # None if not available

    # Idle power: RAPL energy snapshot is instantaneous; skip for now (need two readings)
    m["idle_power_w"] = None

    return m


# ---------------------------------------------------------------------------
# Real measurement — Windows (SSH)
# ---------------------------------------------------------------------------

# Python may or may not be available; use it first then fall back to PowerShell
_WIN_CPU_PY = (
    'python -c "import time; n=500000; t=time.perf_counter(); '
    '[i*(i+1) for i in range(n)]; print(n/(time.perf_counter()-t))"'
)
_WIN_CPU_PS = (
    'powershell -Command "$n=500000; $s=[Diagnostics.Stopwatch]::StartNew(); '
    '0..$n | % {$null=$_*($_+1)}; $s.Stop(); Write-Output ($n/$s.Elapsed.TotalSeconds)"'
)
_WIN_IO_PY = (
    'python -c "import os,tempfile,time; d=os.urandom(1048576); '
    'f=tempfile.NamedTemporaryFile(delete=True); t=time.perf_counter(); '
    'f.write(d); f.flush(); os.fsync(f.fileno()); print(1/(time.perf_counter()-t))"'
)
_WIN_BOOT_PS = (
    "powershell -Command \"$b=(gcim Win32_OperatingSystem).LastBootUpTime; "
    "Write-Output ((Get-Date)-$b).TotalSeconds\""
)
_WIN_LATENCY_PS = (
    "powershell -Command \"$s=[Diagnostics.Stopwatch]::StartNew(); "
    "1..10000 | % {$null=$_}; $s.Stop(); Write-Output $s.Elapsed.TotalMilliseconds\""
)


def _measure_windows_real(session: DeviceSession, device_id: str) -> dict[str, Any]:
    m: dict[str, Any] = {}

    r = session.run_payload(_WIN_CPU_PY)
    if not r.success or not (r.stdout or "").strip():
        r = session.run_payload(_WIN_CPU_PS)
    m["cpu_score"] = _parse_float(r.stdout) or 0.0

    r = session.run_payload(_WIN_IO_PY)
    m["io_mb_s"] = _parse_float(r.stdout) or 0.0

    r = session.run_payload(_WIN_BOOT_PS)
    m["boot_time_s"] = _parse_float(r.stdout)

    r = session.run_payload(_WIN_LATENCY_PS)
    m["latency_ms"] = _parse_float(r.stdout) or 0.0

    m["idle_power_w"] = None  # Not available without WMI/hardware monitoring

    return m


# ---------------------------------------------------------------------------
# Real measurement — Android (ADB)
# ---------------------------------------------------------------------------

_ANDROID_CPU_CMD = (
    "python3 -c 'import time; n=200000; t=time.perf_counter(); "
    "[i*(i+1) for i in range(n)]; print(n/(time.perf_counter()-t))' 2>/dev/null || "
    # Shell arithmetic fallback: time a tight loop
    "time (for i in $(seq 50000); do : ; done) 2>&1 | grep real || echo 0"
)
_ANDROID_IO_CMD = (
    "dd if=/dev/zero of=/data/local/tmp/progenitor_bench bs=1048576 count=5 2>&1 && "
    "rm -f /data/local/tmp/progenitor_bench"
)
_ANDROID_BATTERY_CMD = "dumpsys battery 2>/dev/null | grep -E 'level:|status:'"
_ANDROID_FPS_CMD = (
    "dumpsys display 2>/dev/null | grep -i 'mRefreshRate\\|refreshRate' | head -3"
)
_ANDROID_GOV_CMD = "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"
_ANDROID_ANIM_CMD = (
    "settings get global window_animation_scale 2>/dev/null; "
    "settings get global transition_animation_scale 2>/dev/null"
)


def _parse_android_fps(output: str) -> float:
    m = re.search(r"(\d+\.?\d*)\s*(?:Hz|fps|mRefreshRate)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # dumpsys display: "mRefreshRate=60.000004"
    m = re.search(r"mRefreshRate=(\d+\.?\d*)", output)
    if m:
        return float(m.group(1))
    return 60.0  # safe default


def _parse_android_battery(output: str) -> float | None:
    m = re.search(r"level:\s*(\d+)", output)
    return float(m.group(1)) if m else None


def _parse_android_io(output: str) -> float:
    """dd output: '5242880 bytes (5.0 MB) copied, 0.012 s, 437 MB/s'"""
    m = re.search(r"(\d+\.?\d*)\s*MB/s", output)
    if m:
        return float(m.group(1))
    # Alternative format: "5+0 records in ... X bytes/s"
    m = re.search(r"(\d+\.?\d*)\s*GB/s", output)
    if m:
        return float(m.group(1)) * 1024
    return 0.0


def _measure_android_real(session: DeviceSession, device_id: str) -> dict[str, Any]:
    m: dict[str, Any] = {}

    # CPU
    r = session.run_payload(_ANDROID_CPU_CMD)
    cpu = _parse_float(r.stdout) or 0.0
    # If we got a real Python score, it's absolute; scale to be comparable with desktop
    m["cpu_score"] = cpu

    # IO
    r = session.run_payload(_ANDROID_IO_CMD)
    m["io_mb_s"] = _parse_android_io(r.stdout + r.stderr)

    # Display refresh rate
    r = session.run_payload(_ANDROID_FPS_CMD)
    m["frame_rate_fps"] = _parse_android_fps(r.stdout) if r.success else 60.0

    # Battery level (snapshot; drain rate needs two timed readings)
    r = session.run_payload(_ANDROID_BATTERY_CMD)
    level = _parse_android_battery(r.stdout) if r.success else None
    # We expose battery_drain_per_hr as None when we only have a snapshot
    m["battery_drain_per_hr"] = None
    m["battery_level_pct"] = level  # stored in raw

    # Latency: small loop via shell
    r = session.run_payload(
        "python3 -c 'import time; t=time.perf_counter(); "
        "[i for i in range(5000)]; print((time.perf_counter()-t)*1000)' 2>/dev/null || echo 18"
    )
    m["latency_ms"] = _parse_float(r.stdout) or 18.0

    return m


# ---------------------------------------------------------------------------
# Dispatch: choose real or mock path based on session type
# ---------------------------------------------------------------------------

def _is_mock(session: DeviceSession | None) -> bool:
    return session is None or isinstance(session, MockDeviceSession)


def _measure_real(
    device_id: str, session: DeviceSession, device_type: DeviceType
) -> dict[str, Any]:
    """Run actual benchmark commands on the device."""
    if device_type == DeviceType.PC_LINUX:
        return _measure_linux_real(session, device_id)
    if device_type == DeviceType.PC_WINDOWS:
        return _measure_windows_real(session, device_id)
    if device_type == DeviceType.PHONE_ANDROID:
        return _measure_android_real(session, device_id)
    # Unknown: try Linux commands as best-effort
    return _measure_linux_real(session, device_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_baseline(
    device_id: str,
    session: DeviceSession | None = None,
    device_type: DeviceType | None = None,
) -> DeviceBaseline:
    """
    Measure baseline metrics.
    - Real session (SSHSession / ADBSession): runs commands on the device.
    - Mock session / no session: uses local benchmarks (dry-run safe).
    """
    dtype = device_type or _infer_device_type(device_id)

    if not _is_mock(session):
        assert session is not None
        m = _measure_real(device_id, session, dtype)
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
            raw={**m, "device_id": device_id},
        )

    # Mock / dry-run path
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
    Measure post-enhancement metrics.
    - Real session: runs the same benchmark suite again on the device.
    - Mock session: derives simulated improvements from baseline_raw.
    """
    dtype = device_type or _infer_device_type(device_id)

    if not _is_mock(session):
        assert session is not None
        m = _measure_real(device_id, session, dtype)
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
            raw={**m, "device_id": device_id},
            applied_changes=applied_changes or [],
        )

    # Mock / dry-run path
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

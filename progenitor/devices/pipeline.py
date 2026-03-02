"""Enhancement pipeline: measure baseline → apply enhancements → measure after."""

from __future__ import annotations

from progenitor.devices.adapter import AccessAdapter, DeviceSession, MockDeviceSession
from progenitor.devices.types import DeviceBaseline, DeviceAfter, DeviceReport, DeviceType
from progenitor.devices.measure import measure_baseline, measure_after
from progenitor.devices.enhance import apply_enhancements


def run_pipeline(
    device_id: str,
    adapter: AccessAdapter | None = None,
    *,
    power_profile: bool = False,
    cpu_governor: bool = False,
    reduce_animations: bool = False,
    device_type: DeviceType | None = None,
) -> DeviceReport:
    """
    Run the full pipeline: establish session → measure baseline → apply enhancements
    → measure after → build report. Uses mock adapter if none provided.
    """
    from progenitor.devices.adapter import get_default_adapter
    adp = adapter or get_default_adapter()
    session = adp.establish(device_id)
    try:
        baseline = measure_baseline(device_id, session=session, device_type=device_type)
        dtype = baseline.device_type

        applied = apply_enhancements(
            dtype,
            session,
            power_profile=power_profile,
            cpu_governor=cpu_governor,
            reduce_animations=reduce_animations,
        )

        after = measure_after(
            device_id,
            session=session,
            device_type=dtype,
            applied_changes=applied,
        )

        # Speedup ratios (avoid div by zero)
        speedup_cpu = (after.cpu_score / baseline.cpu_score) if baseline.cpu_score else 1.0
        speedup_io = (after.io_mb_s / baseline.io_mb_s) if baseline.io_mb_s else 1.0
        speedup_latency = (baseline.latency_ms / after.latency_ms) if after.latency_ms else 1.0

        msg = "Enhanced. Applied: " + ", ".join(applied) if applied else "No levers enabled (dry-run or mock)."
        return DeviceReport(
            device_id=device_id,
            device_type=dtype,
            baseline=baseline,
            after=after,
            speedup_cpu=speedup_cpu,
            speedup_io=speedup_io,
            speedup_latency=speedup_latency,
            message=msg,
        )
    finally:
        session.close()

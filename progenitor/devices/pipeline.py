"""Enhancement pipeline: measure baseline → apply enhancements → measure after."""

from __future__ import annotations

from progenitor.devices.adapter import AccessAdapter
from progenitor.devices.types import DeviceReport, DeviceType
from progenitor.devices.measure import measure_baseline, measure_after
from progenitor.devices.enhance import EnhanceOptions, LeverResult, apply_enhancements


def run_pipeline(
    device_id: str,
    adapter: AccessAdapter | None = None,
    opts: EnhanceOptions | None = None,
    device_type: DeviceType | None = None,
) -> DeviceReport:
    """
    Run the full pipeline:
      establish session → measure baseline → apply enhancements → measure after → report.

    Uses mock adapter if none provided (safe for dry-run and tests).
    """
    from progenitor.devices.adapter import get_default_adapter
    adp = adapter or get_default_adapter()
    session = adp.establish(device_id)
    if opts is None:
        opts = EnhanceOptions()

    try:
        baseline = measure_baseline(device_id, session=session, device_type=device_type)
        dtype = baseline.device_type

        lever_results: list[LeverResult] = apply_enhancements(dtype, session, opts)
        applied_changes = [str(r) for r in lever_results]

        after = measure_after(
            device_id,
            session=session,
            device_type=dtype,
            applied_changes=applied_changes,
            baseline_raw=baseline.raw,
        )

        def _ratio(a: float, b: float) -> float:
            return a / b if b else 1.0

        speedup_cpu = _ratio(after.cpu_score, baseline.cpu_score)
        speedup_io = _ratio(after.io_mb_s, baseline.io_mb_s)
        speedup_latency = _ratio(baseline.latency_ms, after.latency_ms)  # lower latency → ratio > 1
        speedup_fps = _ratio(after.frame_rate_fps or 0, baseline.frame_rate_fps or 0) if baseline.frame_rate_fps else 1.0
        improvement_boot = (baseline.boot_time_s or 0) - (after.boot_time_s or 0)
        improvement_battery = (baseline.battery_drain_per_hr or 0) - (after.battery_drain_per_hr or 0)

        msg = "Enhanced. Applied: " + ", ".join(applied_changes) if applied_changes else "No levers enabled. Use --help for available levers."
        return DeviceReport(
            device_id=device_id,
            device_type=dtype,
            baseline=baseline,
            after=after,
            speedup_cpu=speedup_cpu,
            speedup_io=speedup_io,
            speedup_latency=speedup_latency,
            speedup_frame_rate=speedup_fps,
            improvement_boot_s=improvement_boot,
            improvement_battery_pct=improvement_battery,
            message=msg,
        )
    finally:
        session.close()

"""Tests for Phase 3 devices: pipeline, adapter, measure, enhance."""

import pytest

from progenitor.devices import (
    mock_adapter,
    run_pipeline,
    DeviceType,
    DeviceBaseline,
    DeviceReport,
    EnhanceOptions,
    LeverResult,
)
from progenitor.devices.adapter import MockDeviceSession, PayloadResult
from progenitor.devices.measure import measure_baseline, measure_after, _mock_metrics_for, _mock_metrics_after
from progenitor.devices.enhance import apply_enhancements


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def test_mock_adapter_list_devices() -> None:
    adapter = mock_adapter()
    devices = adapter.list_devices()
    assert "127.0.0.1" in devices
    assert "mock-device" in devices


def test_mock_adapter_establish_returns_mock_session() -> None:
    adapter = mock_adapter()
    session = adapter.establish("192.168.1.42")
    assert isinstance(session, MockDeviceSession)
    result = session.run_payload("measure_baseline")
    assert result.success
    session.close()


# ---------------------------------------------------------------------------
# Measure — device-type-specific mock metrics
# ---------------------------------------------------------------------------

def test_mock_metrics_android_have_fps_and_battery() -> None:
    m = _mock_metrics_for(DeviceType.PHONE_ANDROID)
    assert m["frame_rate_fps"] > 0
    assert m["battery_drain_per_hr"] > 0
    assert "boot_time_s" not in m


def test_mock_metrics_windows_have_boot_and_power() -> None:
    m = _mock_metrics_for(DeviceType.PC_WINDOWS)
    assert m["boot_time_s"] > 0
    assert m["idle_power_w"] > 0
    assert "frame_rate_fps" not in m


def test_mock_metrics_linux_have_boot_and_power() -> None:
    m = _mock_metrics_for(DeviceType.PC_LINUX)
    assert m["boot_time_s"] > 0
    assert m["idle_power_w"] > 0


def test_mock_after_android_improves_fps_and_latency() -> None:
    base = _mock_metrics_for(DeviceType.PHONE_ANDROID)
    after = _mock_metrics_after(DeviceType.PHONE_ANDROID, base)
    assert after["frame_rate_fps"] >= base["frame_rate_fps"]
    assert after["latency_ms"] < base["latency_ms"]
    assert after["cpu_score"] > base["cpu_score"]


def test_mock_after_linux_improves_io_and_cpu() -> None:
    base = _mock_metrics_for(DeviceType.PC_LINUX)
    after = _mock_metrics_after(DeviceType.PC_LINUX, base)
    assert after["io_mb_s"] > base["io_mb_s"]
    assert after["cpu_score"] > base["cpu_score"]


def test_mock_after_windows_improves_cpu_and_latency() -> None:
    base = _mock_metrics_for(DeviceType.PC_WINDOWS)
    after = _mock_metrics_after(DeviceType.PC_WINDOWS, base)
    assert after["cpu_score"] > base["cpu_score"]
    assert after["latency_ms"] < base["latency_ms"]


def test_measure_baseline_android_mock_fields() -> None:
    session = MockDeviceSession("android-01")
    baseline = measure_baseline("android-01", session=session, device_type=DeviceType.PHONE_ANDROID)
    assert baseline.device_type == DeviceType.PHONE_ANDROID
    assert baseline.frame_rate_fps is not None
    assert baseline.battery_drain_per_hr is not None
    assert baseline.boot_time_s is None


def test_measure_baseline_linux_mock_fields() -> None:
    session = MockDeviceSession("10.0.0.5")
    baseline = measure_baseline("10.0.0.5", session=session, device_type=DeviceType.PC_LINUX)
    assert baseline.device_type == DeviceType.PC_LINUX
    assert baseline.boot_time_s is not None
    assert baseline.idle_power_w is not None
    assert baseline.frame_rate_fps is None


def test_measure_after_propagates_applied_changes() -> None:
    session = MockDeviceSession("10.0.0.5")
    baseline = measure_baseline("10.0.0.5", session=session, device_type=DeviceType.PC_LINUX)
    after = measure_after(
        "10.0.0.5",
        session=session,
        device_type=DeviceType.PC_LINUX,
        applied_changes=["cpu_governor: powersave → performance"],
        baseline_raw=baseline.raw,
    )
    assert "cpu_governor: powersave → performance" in after.applied_changes
    assert after.cpu_score > 0


# ---------------------------------------------------------------------------
# Enhance — per-device-type levers (opt-in only)
# ---------------------------------------------------------------------------

def test_linux_no_levers_returns_empty() -> None:
    results = apply_enhancements(DeviceType.PC_LINUX, None, EnhanceOptions())
    assert results == []


def test_linux_cpu_governor_lever() -> None:
    opts = EnhanceOptions(cpu_governor=True)
    results = apply_enhancements(DeviceType.PC_LINUX, None, opts)
    names = [r.name for r in results]
    assert "cpu_governor" in names
    r = next(r for r in results if r.name == "cpu_governor")
    assert r.after == "performance"


def test_linux_multiple_levers() -> None:
    opts = EnhanceOptions(cpu_governor=True, io_scheduler=True, swappiness=True)
    results = apply_enhancements(DeviceType.PC_LINUX, None, opts)
    names = [r.name for r in results]
    assert "cpu_governor" in names
    assert "io_scheduler" in names
    assert "vm.swappiness" in names
    assert len(results) == 3


def test_windows_power_plan_lever() -> None:
    opts = EnhanceOptions(power_plan=True)
    results = apply_enhancements(DeviceType.PC_WINDOWS, None, opts)
    names = [r.name for r in results]
    assert "power_plan" in names
    r = next(r for r in results if r.name == "power_plan")
    assert r.after == "High Performance"


def test_windows_multiple_levers() -> None:
    opts = EnhanceOptions(power_plan=True, disable_visual_effects=True, game_mode=True)
    results = apply_enhancements(DeviceType.PC_WINDOWS, None, opts)
    names = [r.name for r in results]
    assert "power_plan" in names
    assert "visual_effects" in names
    assert "game_mode" in names


def test_android_reduce_animations_lever() -> None:
    opts = EnhanceOptions(reduce_animations=True)
    results = apply_enhancements(DeviceType.PHONE_ANDROID, None, opts)
    names = [r.name for r in results]
    assert "animations" in names
    r = next(r for r in results if r.name == "animations")
    assert r.after == 0.0


def test_android_all_levers() -> None:
    opts = EnhanceOptions(
        performance_profile=True,
        disable_doze=True,
        reduce_animations=True,
        background_limits=True,
    )
    results = apply_enhancements(DeviceType.PHONE_ANDROID, None, opts)
    names = [r.name for r in results]
    assert "cpu_governor" in names
    assert "doze_mode" in names
    assert "animations" in names
    assert "background_process_limit" in names
    assert len(results) == 4


def test_levers_do_not_cross_device_types() -> None:
    # Windows levers should not appear when device is Linux
    opts = EnhanceOptions(power_plan=True, game_mode=True)
    results = apply_enhancements(DeviceType.PC_LINUX, None, opts)
    assert results == []

    # Linux levers should not appear when device is Android
    opts = EnhanceOptions(io_scheduler=True, swappiness=True)
    results = apply_enhancements(DeviceType.PHONE_ANDROID, None, opts)
    assert results == []


# ---------------------------------------------------------------------------
# Pipeline — end-to-end dry-run per device type
# ---------------------------------------------------------------------------

def test_pipeline_linux_dry_run() -> None:
    opts = EnhanceOptions(cpu_governor=True, io_scheduler=True)
    report = run_pipeline("10.0.0.1", adapter=mock_adapter(), opts=opts, device_type=DeviceType.PC_LINUX)
    assert isinstance(report, DeviceReport)
    assert report.device_type == DeviceType.PC_LINUX
    assert report.speedup_cpu > 1.0
    assert report.speedup_io > 1.0
    assert report.baseline.boot_time_s is not None
    assert report.improvement_boot_s > 0


def test_pipeline_windows_dry_run() -> None:
    opts = EnhanceOptions(power_plan=True, disable_visual_effects=True)
    report = run_pipeline("10.0.0.2", adapter=mock_adapter(), opts=opts, device_type=DeviceType.PC_WINDOWS)
    assert report.device_type == DeviceType.PC_WINDOWS
    assert report.speedup_cpu > 1.0
    assert report.baseline.idle_power_w is not None


def test_pipeline_android_dry_run() -> None:
    opts = EnhanceOptions(performance_profile=True, reduce_animations=True, disable_doze=True)
    report = run_pipeline("192.168.1.10", adapter=mock_adapter(), opts=opts, device_type=DeviceType.PHONE_ANDROID)
    assert report.device_type == DeviceType.PHONE_ANDROID
    assert report.speedup_cpu > 1.0
    assert report.speedup_frame_rate > 1.0
    assert report.speedup_latency > 1.0
    assert report.baseline.frame_rate_fps is not None


def test_pipeline_no_levers_message() -> None:
    report = run_pipeline("127.0.0.1", adapter=mock_adapter(), opts=EnhanceOptions())
    assert "No levers" in report.message
    assert report.after.applied_changes == []

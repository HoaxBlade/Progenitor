"""Tests for Phase 3 devices: pipeline, adapter, measure, enhance."""

import pytest

from progenitor.devices import (
    mock_adapter,
    run_pipeline,
    DeviceType,
    DeviceBaseline,
    DeviceReport,
)
from progenitor.devices.adapter import MockDeviceSession, PayloadResult
from progenitor.devices.measure import measure_baseline, measure_after
from progenitor.devices.enhance import apply_enhancements


def test_mock_adapter_list_devices() -> None:
    adapter = mock_adapter()
    devices = adapter.list_devices()
    assert "127.0.0.1" in devices
    assert "mock-device" in devices


def test_mock_adapter_establish() -> None:
    adapter = mock_adapter()
    session = adapter.establish("192.168.1.1")
    assert isinstance(session, MockDeviceSession)
    result = session.run_payload("measure_baseline")
    assert result.success
    session.close()


def test_measure_baseline_mock() -> None:
    session = MockDeviceSession("127.0.0.1")
    baseline = measure_baseline("127.0.0.1", session=session)
    assert baseline.device_id == "127.0.0.1"
    assert baseline.cpu_score >= 0
    assert baseline.io_mb_s >= 0


def test_apply_enhancements_opt_in() -> None:
    applied = apply_enhancements(DeviceType.PC_LINUX, None, power_profile=True, cpu_governor=False)
    assert "power_profile=performance" in applied
    assert len(applied) == 1


def test_run_pipeline_dry_run() -> None:
    report = run_pipeline("127.0.0.1", adapter=mock_adapter(), power_profile=True)
    assert isinstance(report, DeviceReport)
    assert report.device_id == "127.0.0.1"
    assert report.baseline.cpu_score >= 0
    assert report.after.applied_changes
    assert "power_profile=performance" in report.after.applied_changes

"""Tests for Phase 2 software: measure, analyze, proxy, enhance_software_by_url."""

import pytest

from progenitor.software.measure import SiteMeasurement
from progenitor.software.analyze import analyze, Finding, EnhanceReport
from progenitor.software.proxy import ProgenitorProxy


def _mock_measurement(
    compressed: bool = False,
    cache_control: str = "",
    etag: bool = False,
    size_bytes: int = 50000,
    server: str = "nginx/1.18",
    powered_by: str = "",
) -> SiteMeasurement:
    return SiteMeasurement(
        url="https://example.com/",
        ttfb_ms=100.0,
        total_ms=200.0,
        size_bytes=size_bytes,
        compressed=compressed,
        cache_control=cache_control,
        etag=etag,
        status_code=200,
        server=server,
        powered_by=powered_by,
        p99_ms=300.0,
        raw_times_ms=[200.0],
        api_timings={},
    )


def test_analyze_returns_findings_for_uncompressed() -> None:
    m = _mock_measurement(compressed=False, size_bytes=2 * 1024 * 1024)
    report = analyze(m, "latency")
    assert isinstance(report, EnhanceReport)
    assert report.url == m.url
    assert report.target == "latency"
    assert report.before == m
    assert len(report.findings) >= 1
    compression_findings = [f for f in report.findings if f.dimension == "compression"]
    assert len(compression_findings) == 1
    assert "2.0 MB" in compression_findings[0].detail or "2.0 MB" in str(compression_findings[0].fix_commands)


def test_analyze_findings_have_fix_commands() -> None:
    m = _mock_measurement(compressed=False, cache_control="no-cache", size_bytes=10000)
    report = analyze(m, "all")
    assert report.findings
    for f in report.findings:
        assert isinstance(f, Finding)
        assert f.detail
        assert isinstance(f.fix_commands, list)
        assert len(f.fix_commands) >= 1


def test_analyze_nginx_stack_gets_nginx_commands() -> None:
    m = _mock_measurement(compressed=False, server="nginx/1.18", size_bytes=50000)
    report = analyze(m, "latency")
    compression = next((f for f in report.findings if f.dimension == "compression"), None)
    assert compression is not None
    assert any("nginx" in line.lower() for line in compression.fix_commands)


def test_analyze_no_findings_when_healthy() -> None:
    m = _mock_measurement(compressed=True, cache_control="max-age=3600", etag=True, size_bytes=100)
    report = analyze(m, "latency")
    assert len(report.findings) == 0
    assert "No major issues" in report.estimated_speedup or "No issues" in report.estimated_speedup


def test_proxy_start_stop() -> None:
    proxy = ProgenitorProxy(origin="https://example.com", port=0)
    port = proxy.start()
    assert port > 0
    assert proxy.base_url() == f"http://127.0.0.1:{port}"
    proxy.stop()
    # After stop, base_url might still return the old port; we just check stop doesn't raise
    assert True

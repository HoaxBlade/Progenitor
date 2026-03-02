"""
Access adapter: plug point for the external access module.

This repo does not implement the access mechanism. The adapter interface
allows the operator CLI to work with a mock (local) adapter for development
or a real adapter that invokes the external access module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from progenitor.devices.types import DeviceBaseline, DeviceAfter, DeviceType


@dataclass
class PayloadResult:
    """Result of running a payload on the device (or locally)."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] | None = None  # e.g. baseline or after metrics
    error: str | None = None


class DeviceSession(ABC):
    """Abstract session to a device. The access module provides a concrete implementation."""

    @abstractmethod
    def run_payload(self, payload: str | list[str], env: dict[str, str] | None = None) -> PayloadResult:
        """
        Run a command or script on the device. payload can be a single command string
        or a list of commands. env is optional environment for the payload.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the session."""
        ...


class AccessAdapter(ABC):
    """Adapter to establish a session to a device. Default: mock (local). Real: calls access module."""

    @abstractmethod
    def list_devices(self) -> list[str]:
        """Discover devices on the LAN (e.g. IPs or device IDs). Return empty if not supported."""
        ...

    @abstractmethod
    def establish(self, device_id: str) -> DeviceSession:
        """Establish a session to the given device. device_id is e.g. IP, hostname, or device ID."""
        ...


class MockDeviceSession(DeviceSession):
    """Session that runs payload locally (for development / dry-run)."""

    def __init__(self, device_id: str, device_type: DeviceType = DeviceType.UNKNOWN):
        self._device_id = device_id
        self._device_type = device_type

    def run_payload(self, payload: str | list[str], env: dict[str, str] | None = None) -> PayloadResult:
        # In mock mode we don't actually run remote commands; the pipeline will use local measure/enhance
        return PayloadResult(
            success=True,
            stdout=f"[mock] payload would run on device {self._device_id}",
            metrics={"mock": True, "device_id": self._device_id},
        )

    def close(self) -> None:
        pass


class MockAccessAdapter(AccessAdapter):
    """Adapter that never talks to a real device. For --dry-run and tests."""

    def list_devices(self) -> list[str]:
        return ["127.0.0.1", "mock-device"]

    def establish(self, device_id: str) -> DeviceSession:
        return MockDeviceSession(device_id)


def mock_adapter() -> AccessAdapter:
    """Return the default mock adapter."""
    return MockAccessAdapter()


def get_default_adapter() -> AccessAdapter:
    """
    Return the default adapter. If PROGENITOR_ACCESS_MODULE is set, could load
    an external adapter; for now always returns the mock adapter.
    """
    import os
    mod = os.environ.get("PROGENITOR_ACCESS_MODULE")
    if mod:
        # Optional: load adapter from env-specified module (e.g. "my_module:get_adapter")
        # For now we don't implement dynamic load; just return mock
        pass
    return MockAccessAdapter()

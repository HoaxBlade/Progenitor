"""
ADB transport for Android devices.

Prerequisites:
  - Android Debug Bridge installed: https://developer.android.com/tools/adb
    macOS: `brew install android-platform-tools`
    Linux: `sudo apt install adb`
    Windows: install Android SDK Platform Tools
  - USB debugging enabled on the phone:
    Settings → About Phone → tap Build Number 7× → Developer Options → USB Debugging ON
  - For wireless ADB (Android 11+):
    Developer Options → Wireless Debugging → Pair device / connect
    Or: `adb tcpip 5555` then `adb connect <device_ip>:5555`
  - Levers that change CPU governor or disable Doze require root (su) or a
    device with unlocked bootloader. Check output — non-root devices will get
    Permission denied on sysfs writes.

Usage:
    # USB (single connected device)
    adapter = ADBAdapter()
    session = adapter.establish("usb")

    # Wireless / specific serial
    adapter = ADBAdapter(serial="192.168.1.10:5555")
    session = adapter.establish("192.168.1.10")
"""

from __future__ import annotations

import subprocess
from typing import Any

from progenitor.devices.adapter import AccessAdapter, DeviceSession, PayloadResult
from progenitor.devices.types import DeviceType


def _adb(*args: str, serial: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run an adb command, optionally targeting a specific serial."""
    base = ["adb"]
    if serial:
        base += ["-s", serial]
    return subprocess.run(base + list(args), capture_output=True, text=True, timeout=timeout)


class ADBSession(DeviceSession):
    """Session backed by `adb shell` commands."""

    def __init__(self, device_id: str, serial: str | None = None) -> None:
        self._device_id = device_id
        self._serial = serial  # adb serial (e.g. "192.168.1.10:5555" or "emulator-5554")

    @property
    def serial(self) -> str | None:
        return self._serial

    def run_payload(
        self,
        payload: str | list[str],
        env: dict[str, str] | None = None,
    ) -> PayloadResult:
        """Run a shell command on the device via `adb shell`."""
        cmd = " && ".join(payload) if isinstance(payload, list) else payload
        try:
            r = _adb("shell", cmd, serial=self._serial)
            # adb shell returns 0 even on device errors; check for "error:" in output
            failed = r.returncode != 0 or r.stderr.startswith("error:")
            return PayloadResult(
                success=not failed,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        except FileNotFoundError:
            return PayloadResult(
                success=False,
                error="adb not found. Install: brew install android-platform-tools (macOS) or apt install adb (Linux)",
            )
        except subprocess.TimeoutExpired:
            return PayloadResult(success=False, error="ADB command timed out after 60s")
        except Exception as e:
            return PayloadResult(success=False, error=str(e))

    def close(self) -> None:
        pass  # no persistent connection


class ADBAdapter(AccessAdapter):
    """
    Adapter that discovers and connects to Android devices via ADB.

    If serial is given, always use that device. Otherwise use the first
    connected device (USB or TCP).
    """

    def __init__(self, serial: str | None = None) -> None:
        self._serial = serial

    def list_devices(self) -> list[str]:
        """Return list of connected ADB device serials."""
        try:
            r = _adb("devices", timeout=10)
            lines = r.stdout.strip().splitlines()
            devices = []
            for line in lines[1:]:  # skip "List of devices attached"
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices
        except Exception:
            return []

    def establish(self, device_id: str) -> ADBSession:
        """Return an ADB session for the device. Connects over TCP if needed."""
        serial = self._serial
        if serial is None:
            # If device_id looks like an IP, try connecting over TCP
            if _looks_like_ip(device_id):
                addr = device_id if ":" in device_id else f"{device_id}:5555"
                try:
                    r = _adb("connect", addr, timeout=10)
                    if "connected" in r.stdout.lower():
                        serial = addr
                except Exception:
                    pass
            # Fall back to first available device (USB or already connected TCP)
            if serial is None:
                devices = self.list_devices()
                serial = devices[0] if devices else None
        return ADBSession(device_id, serial=serial)


def _looks_like_ip(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() for p in parts)

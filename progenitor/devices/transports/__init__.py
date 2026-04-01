"""Real device transports: SSH (Linux/Windows) and ADB (Android)."""

from progenitor.devices.transports.ssh import SSHAdapter, SSHSession
from progenitor.devices.transports.adb import ADBAdapter, ADBSession

__all__ = ["SSHAdapter", "SSHSession", "ADBAdapter", "ADBSession"]

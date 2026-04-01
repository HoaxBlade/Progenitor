"""Real device transports: SSH (Linux/Windows), ADB (Android), Agent (any device)."""

from progenitor.devices.transports.ssh import SSHAdapter, SSHSession
from progenitor.devices.transports.adb import ADBAdapter, ADBSession
from progenitor.devices.transports.agent import AgentAdapter, AgentSession

__all__ = [
    "SSHAdapter", "SSHSession",
    "ADBAdapter", "ADBSession",
    "AgentAdapter", "AgentSession",
]

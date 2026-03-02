"""
Apply opt-in enhancements to a device. Device-type specific; all changes reversible.
"""

from __future__ import annotations

from progenitor.devices.types import DeviceType
from progenitor.devices.adapter import DeviceSession, MockDeviceSession


def apply_enhancements(
    device_type: DeviceType,
    session: DeviceSession | None,
    *,
    power_profile: bool = False,
    cpu_governor: bool = False,
    reduce_animations: bool = False,
) -> list[str]:
    """
    Apply only the levers you enable. Returns list of applied change descriptions.
    In this repo we do not implement real device changes; the access module
    would run the actual tuning on the device. Here we return a mock list
    when running locally (MockDeviceSession).
    """
    applied: list[str] = []
    if power_profile:
        applied.append("power_profile=performance")
    if cpu_governor:
        applied.append("cpu_governor=performance")
    if reduce_animations:
        applied.append("reduce_animations=true")

    if session is not None and not isinstance(session, MockDeviceSession):
        # Real session: send payload to apply these levers (access module runs it)
        payload = ["apply_enhancements"] + applied
        session.run_payload(payload)

    return applied

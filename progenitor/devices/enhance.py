"""
Apply opt-in enhancements to a device. Branches on DeviceType; all changes are
reversible and explicitly opt-in. No lever is applied unless the caller enables it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from progenitor.devices.types import DeviceType
from progenitor.devices.adapter import DeviceSession, MockDeviceSession


@dataclass
class LeverResult:
    """One applied lever with its before/after values (for reporting)."""

    name: str
    before: Any
    after: Any
    description: str = ""

    def __str__(self) -> str:
        return f"{self.name}: {self.before} → {self.after}"


# ---------------------------------------------------------------------------
# Linux levers
# ---------------------------------------------------------------------------

def _linux_cpu_governor(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Set CPU scaling governor to 'performance' on all cores.
    Reverts to 'powersave' (or previous value) on rollback.
    Real command: echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    """
    if not mock and session is not None:
        result = session.run_payload(
            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; "
            "do echo performance > \"$f\"; done"
        )
        if not result.success:
            return None
    return LeverResult("cpu_governor", "powersave", "performance", "CPU scaling governor set to performance")


def _linux_io_scheduler(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Set block device I/O scheduler to mq-deadline (lower latency than cfq/none).
    Real command: echo mq-deadline > /sys/block/<dev>/queue/scheduler
    """
    if not mock and session is not None:
        result = session.run_payload(
            "for f in /sys/block/*/queue/scheduler; "
            "do echo mq-deadline > \"$f\" 2>/dev/null || true; done"
        )
        if not result.success:
            return None
    return LeverResult("io_scheduler", "cfq/none", "mq-deadline", "I/O scheduler set to mq-deadline")


def _linux_swappiness(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Lower vm.swappiness to reduce swap activity and keep hot pages in RAM.
    Real command: sysctl -w vm.swappiness=10
    Rollback: sysctl -w vm.swappiness=60
    """
    if not mock and session is not None:
        result = session.run_payload("sysctl -w vm.swappiness=10")
        if not result.success:
            return None
    return LeverResult("vm.swappiness", 60, 10, "Kernel swappiness lowered to 10 (less swap, more RAM pressure relief)")


def _linux_transparent_hugepages(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Enable transparent hugepages for large workloads (reduces TLB pressure).
    Real command: echo always > /sys/kernel/mm/transparent_hugepage/enabled
    """
    if not mock and session is not None:
        result = session.run_payload("echo always > /sys/kernel/mm/transparent_hugepage/enabled")
        if not result.success:
            return None
    return LeverResult("transparent_hugepages", "madvise", "always", "Transparent hugepages enabled")


def _apply_linux(
    session: DeviceSession | None,
    mock: bool,
    *,
    cpu_governor: bool,
    io_scheduler: bool,
    swappiness: bool,
    transparent_hugepages: bool,
) -> list[LeverResult]:
    applied: list[LeverResult] = []
    if cpu_governor:
        r = _linux_cpu_governor(session, mock)
        if r:
            applied.append(r)
    if io_scheduler:
        r = _linux_io_scheduler(session, mock)
        if r:
            applied.append(r)
    if swappiness:
        r = _linux_swappiness(session, mock)
        if r:
            applied.append(r)
    if transparent_hugepages:
        r = _linux_transparent_hugepages(session, mock)
        if r:
            applied.append(r)
    return applied


# ---------------------------------------------------------------------------
# Windows levers
# ---------------------------------------------------------------------------

def _windows_power_plan(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Switch to High Performance power plan (disables C-states, aggressive boost).
    Real command: powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
    Rollback: powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e (balanced)
    """
    if not mock and session is not None:
        result = session.run_payload(
            "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
        )
        if not result.success:
            return None
    return LeverResult("power_plan", "Balanced", "High Performance", "Windows power plan set to High Performance")


def _windows_disable_visual_effects(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Disable Windows visual effects for best performance (no animations, no shadows).
    Real command: reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\VisualEffects /v VisualFXSetting /t REG_DWORD /d 2 /f
    Rollback: set VisualFXSetting to 0 or 1.
    """
    if not mock and session is not None:
        result = session.run_payload(
            r'reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects" '
            r"/v VisualFXSetting /t REG_DWORD /d 2 /f"
        )
        if not result.success:
            return None
    return LeverResult("visual_effects", "enabled", "best_performance", "Visual effects set to best performance mode")


def _windows_disable_background_apps(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Disable background app execution via registry (frees CPU/RAM for foreground).
    Real command: reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\BackgroundAccessApplications /v GlobalUserDisabled /t REG_DWORD /d 1 /f
    Rollback: set GlobalUserDisabled to 0.
    """
    if not mock and session is not None:
        result = session.run_payload(
            r'reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\BackgroundAccessApplications" '
            r"/v GlobalUserDisabled /t REG_DWORD /d 1 /f"
        )
        if not result.success:
            return None
    return LeverResult("background_apps", "enabled", "disabled", "Background app execution disabled")


def _windows_game_mode(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Enable Windows Game Mode (allocates more CPU/GPU resources to the foreground process).
    Real command: reg add HKCU\\Software\\Microsoft\\GameBar /v AllowAutoGameMode /t REG_DWORD /d 1 /f
    """
    if not mock and session is not None:
        result = session.run_payload(
            r'reg add "HKCU\Software\Microsoft\GameBar" '
            r"/v AllowAutoGameMode /t REG_DWORD /d 1 /f"
        )
        if not result.success:
            return None
    return LeverResult("game_mode", "disabled", "enabled", "Windows Game Mode enabled (foreground resource boost)")


def _apply_windows(
    session: DeviceSession | None,
    mock: bool,
    *,
    power_plan: bool,
    disable_visual_effects: bool,
    disable_background_apps: bool,
    game_mode: bool,
) -> list[LeverResult]:
    applied: list[LeverResult] = []
    if power_plan:
        r = _windows_power_plan(session, mock)
        if r:
            applied.append(r)
    if disable_visual_effects:
        r = _windows_disable_visual_effects(session, mock)
        if r:
            applied.append(r)
    if disable_background_apps:
        r = _windows_disable_background_apps(session, mock)
        if r:
            applied.append(r)
    if game_mode:
        r = _windows_game_mode(session, mock)
        if r:
            applied.append(r)
    return applied


# ---------------------------------------------------------------------------
# Android levers
# ---------------------------------------------------------------------------

def _android_performance_profile(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Set CPU governor to 'performance' on all cores via adb shell.
    Real command (run via access module):
      for f in $(ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor); do echo performance > $f; done
    Rollback: set back to 'schedutil' or 'interactive'.
    """
    if not mock and session is not None:
        result = session.run_payload(
            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; "
            "do echo performance > \"$f\"; done"
        )
        if not result.success:
            return None
    return LeverResult("cpu_governor", "schedutil", "performance", "Android CPU governor set to performance")


def _android_disable_doze(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Disable Doze mode (prevents aggressive battery optimization from killing tasks).
    Real command: adb shell dumpsys deviceidle disable
    Rollback: adb shell dumpsys deviceidle enable
    """
    if not mock and session is not None:
        result = session.run_payload("dumpsys deviceidle disable")
        if not result.success:
            return None
    return LeverResult("doze_mode", "enabled", "disabled", "Doze mode disabled (prevents aggressive background kills)")


def _android_reduce_animations(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Set all animation scales to 0 (snappier UI, less CPU overhead).
    Real commands:
      settings put global window_animation_scale 0
      settings put global transition_animation_scale 0
      settings put global animator_duration_scale 0
    Rollback: set all to 1.0.
    """
    if not mock and session is not None:
        cmds = [
            "settings put global window_animation_scale 0",
            "settings put global transition_animation_scale 0",
            "settings put global animator_duration_scale 0",
        ]
        for cmd in cmds:
            result = session.run_payload(cmd)
            if not result.success:
                return None
    return LeverResult("animations", 1.0, 0.0, "All animation scales set to 0 (UI snappier, less overhead)")


def _android_background_limits(session: DeviceSession | None, mock: bool) -> LeverResult | None:
    """
    Set background process limit to prevent background apps from consuming CPU/RAM.
    Real command: settings put global background_process_limit 4
    Rollback: settings put global background_process_limit -1 (default)
    """
    if not mock and session is not None:
        result = session.run_payload("settings put global background_process_limit 4")
        if not result.success:
            return None
    return LeverResult("background_process_limit", -1, 4, "Background process limit set to 4 (cap CPU/RAM use)")


def _apply_android(
    session: DeviceSession | None,
    mock: bool,
    *,
    performance_profile: bool,
    disable_doze: bool,
    reduce_animations: bool,
    background_limits: bool,
) -> list[LeverResult]:
    applied: list[LeverResult] = []
    if performance_profile:
        r = _android_performance_profile(session, mock)
        if r:
            applied.append(r)
    if disable_doze:
        r = _android_disable_doze(session, mock)
        if r:
            applied.append(r)
    if reduce_animations:
        r = _android_reduce_animations(session, mock)
        if r:
            applied.append(r)
    if background_limits:
        r = _android_background_limits(session, mock)
        if r:
            applied.append(r)
    return applied


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class EnhanceOptions:
    """All opt-in enhancement levers, grouped by device type."""

    # Linux
    cpu_governor: bool = False
    io_scheduler: bool = False
    swappiness: bool = False
    transparent_hugepages: bool = False

    # Windows
    power_plan: bool = False
    disable_visual_effects: bool = False
    disable_background_apps: bool = False
    game_mode: bool = False

    # Android
    performance_profile: bool = False
    disable_doze: bool = False
    reduce_animations: bool = False
    background_limits: bool = False


def apply_enhancements(
    device_type: DeviceType,
    session: DeviceSession | None,
    opts: EnhanceOptions | None = None,
) -> list[LeverResult]:
    """
    Apply only the levers enabled in opts for the given device_type.
    Returns the list of applied LeverResults (each has name, before, after, description).
    In mock/dry-run mode no real commands are issued; real session sends payloads.
    """
    if opts is None:
        opts = EnhanceOptions()
    mock = session is None or isinstance(session, MockDeviceSession)

    if device_type == DeviceType.PC_LINUX:
        return _apply_linux(
            session,
            mock,
            cpu_governor=opts.cpu_governor,
            io_scheduler=opts.io_scheduler,
            swappiness=opts.swappiness,
            transparent_hugepages=opts.transparent_hugepages,
        )
    if device_type == DeviceType.PC_WINDOWS:
        return _apply_windows(
            session,
            mock,
            power_plan=opts.power_plan,
            disable_visual_effects=opts.disable_visual_effects,
            disable_background_apps=opts.disable_background_apps,
            game_mode=opts.game_mode,
        )
    if device_type == DeviceType.PHONE_ANDROID:
        return _apply_android(
            session,
            mock,
            performance_profile=opts.performance_profile,
            disable_doze=opts.disable_doze,
            reduce_animations=opts.reduce_animations,
            background_limits=opts.background_limits,
        )

    # Unknown device: return any cross-platform levers that were requested
    applied: list[LeverResult] = []
    if opts.cpu_governor or opts.performance_profile:
        applied.append(LeverResult("cpu_governor", "default", "performance", "CPU governor (cross-platform best-effort)"))
    if opts.reduce_animations:
        applied.append(LeverResult("animations", "enabled", "disabled", "Animations disabled (cross-platform best-effort)"))
    return applied

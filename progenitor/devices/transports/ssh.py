"""
SSH transport for Linux and Windows devices.

Uses paramiko if installed (`pip install paramiko`), otherwise falls back to
subprocess `ssh`. Paramiko is recommended: no TTY, no host-key prompts.

Usage:
    adapter = SSHAdapter("192.168.1.100", user="ubuntu", key_path="~/.ssh/id_rsa")
    session = adapter.establish("192.168.1.100")
    result  = session.run_payload("echo hello")
    session.close()
"""

from __future__ import annotations

import subprocess
import shlex
from pathlib import Path
from typing import Any

from progenitor.devices.adapter import AccessAdapter, DeviceSession, PayloadResult
from progenitor.devices.types import DeviceType


class SSHSession(DeviceSession):
    """Session backed by an SSH connection (paramiko or subprocess ssh)."""

    def __init__(
        self,
        device_id: str,
        device_type: DeviceType,
        *,
        # paramiko client (if available)
        _paramiko_client: Any | None = None,
        # subprocess ssh fallback params
        user: str | None = None,
        port: int = 22,
        key_path: str | None = None,
        password: str | None = None,
    ) -> None:
        self._device_id = device_id
        self._device_type = device_type
        self._client = _paramiko_client
        self._user = user
        self._port = port
        self._key_path = key_path
        self._password = password

    # ------------------------------------------------------------------
    # DeviceSession interface
    # ------------------------------------------------------------------

    def run_payload(
        self,
        payload: str | list[str],
        env: dict[str, str] | None = None,
    ) -> PayloadResult:
        cmd = " && ".join(payload) if isinstance(payload, list) else payload
        if self._client is not None:
            return self._run_paramiko(cmd)
        return self._run_subprocess(cmd)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_paramiko(self, cmd: str) -> PayloadResult:
        try:
            _, stdout, stderr = self._client.exec_command(cmd, timeout=60)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            rc = stdout.channel.recv_exit_status()
            return PayloadResult(success=(rc == 0), stdout=out, stderr=err)
        except Exception as e:
            return PayloadResult(success=False, error=str(e))

    def _run_subprocess(self, cmd: str) -> PayloadResult:
        """Fall back to subprocess `ssh` when paramiko is not installed."""
        ssh_args = ["ssh", "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-p", str(self._port)]
        if self._key_path:
            ssh_args += ["-i", str(Path(self._key_path).expanduser())]
        host = f"{self._user}@{self._device_id}" if self._user else self._device_id
        ssh_args.append(host)
        ssh_args.append(cmd)
        try:
            r = subprocess.run(ssh_args, capture_output=True, text=True, timeout=60)
            return PayloadResult(
                success=(r.returncode == 0),
                stdout=r.stdout,
                stderr=r.stderr,
            )
        except FileNotFoundError:
            return PayloadResult(
                success=False,
                error="ssh not found. Install OpenSSH client or run: pip install paramiko",
            )
        except subprocess.TimeoutExpired:
            return PayloadResult(success=False, error="SSH command timed out after 60s")
        except Exception as e:
            return PayloadResult(success=False, error=str(e))


class SSHAdapter(AccessAdapter):
    """
    Adapter for SSH-accessible devices (Linux and Windows with OpenSSH).

    Windows prerequisites:
      - Enable OpenSSH Server: Settings → Apps → Optional Features → OpenSSH Server
      - Start service: `Start-Service sshd` / `Set-Service -Name sshd -StartupType Automatic`

    Linux prerequisites:
      - OpenSSH server installed: `sudo apt install openssh-server` (or equivalent)
      - Root or sudoer access for governor/scheduler/sysctl levers

    Authentication (in order of preference):
      1. key_path: path to private key (most secure)
      2. password: SSH password (less secure)
      3. Neither: relies on ssh-agent or default ~/.ssh/id_* keys
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        user: str | None = None,
        port: int = 22,
        key_path: str | None = None,
        password: str | None = None,
        device_type: DeviceType = DeviceType.PC_LINUX,
    ) -> None:
        self._default_host = host
        self._user = user
        self._port = port
        self._key_path = key_path
        self._password = password
        self._device_type = device_type

    def list_devices(self) -> list[str]:
        if self._default_host:
            return [self._default_host]
        return []

    def establish(self, device_id: str) -> SSHSession:
        """Connect via SSH. Tries paramiko first, then subprocess ssh."""
        try:
            import paramiko  # type: ignore
            return self._connect_paramiko(device_id, paramiko)
        except ImportError:
            pass
        # Subprocess fallback (no password support without sshpass)
        return SSHSession(
            device_id,
            self._device_type,
            user=self._user,
            port=self._port,
            key_path=self._key_path,
        )

    def _connect_paramiko(self, device_id: str, paramiko: Any) -> SSHSession:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict[str, Any] = {"hostname": device_id, "port": self._port, "timeout": 15}
        if self._user:
            kwargs["username"] = self._user
        if self._password:
            kwargs["password"] = self._password
        if self._key_path:
            kwargs["key_filename"] = str(Path(self._key_path).expanduser())
        client.connect(**kwargs)
        return SSHSession(device_id, self._device_type, _paramiko_client=client)

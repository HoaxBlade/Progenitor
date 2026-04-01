"""
Agent transport — operator side.

Connects to a Progenitor Agent running on the target device.
The agent listens on a TCP port; this adapter sends commands and reads results.

Usage:
    adapter = AgentAdapter("192.168.1.10", token="abc123def456")
    session = adapter.establish("192.168.1.10")
    result  = session.run_payload("echo hello")
    print(result.stdout)   # "hello"
    session.close()
"""

from __future__ import annotations

import json
import socket

from progenitor.devices.adapter import AccessAdapter, DeviceSession, PayloadResult
from progenitor.devices.agent.server import DEFAULT_PORT


class AgentSession(DeviceSession):
    """Session that talks to a Progenitor Agent over TCP."""

    def __init__(self, host: str, port: int, token: str) -> None:
        self._host = host
        self._port = port
        self._token = token

    def run_payload(
        self,
        payload: str | list[str],
        env: dict[str, str] | None = None,
    ) -> PayloadResult:
        """Send a command to the agent and return its result."""
        cmd = " && ".join(payload) if isinstance(payload, list) else payload
        req = {"token": self._token, "cmd": cmd}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((self._host, self._port))
            sock.sendall(json.dumps(req).encode() + b"\n")

            # Read until newline (response is a single JSON line)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            sock.close()

            resp = json.loads(data.split(b"\n")[0])
            return PayloadResult(
                success=resp.get("success", False),
                stdout=resp.get("stdout", ""),
                stderr=resp.get("stderr", ""),
                metrics=None,
                error=resp.get("error"),
            )

        except ConnectionRefusedError:
            return PayloadResult(
                success=False,
                error=(
                    f"Connection refused at {self._host}:{self._port}. "
                    "Is the Progenitor Agent running on the target device? "
                    f"Start it with: progenitor agent start --token {self._token}"
                ),
            )
        except socket.timeout:
            return PayloadResult(
                success=False,
                error=f"Timed out connecting to agent at {self._host}:{self._port}",
            )
        except Exception as e:
            return PayloadResult(success=False, error=str(e))

    def close(self) -> None:
        pass  # stateless (new TCP connection per command)


class AgentAdapter(AccessAdapter):
    """
    Adapter that connects to a Progenitor Agent already running on the target.

    The agent is the lightweight daemon the customer installed once. After that,
    any operator with the correct token can reach the device over the LAN with
    no further action from the customer.
    """

    def __init__(
        self,
        token: str,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._token = token
        self._port = port

    def list_devices(self) -> list[str]:
        """
        Agent-based discovery: scan the LAN for devices with the agent port open.
        Returns reachable IPs. Fast scan (0.3s timeout per host).
        """
        import ipaddress
        import concurrent.futures

        # Detect local subnet from default interface
        subnet = _local_subnet()
        if not subnet:
            return []

        reachable: list[str] = []

        def _probe(ip: str) -> str | None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect((ip, self._port))
                s.close()
                return ip
            except Exception:
                return None

        hosts = [str(h) for h in ipaddress.ip_network(subnet, strict=False).hosts()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as ex:
            for result in ex.map(_probe, hosts):
                if result:
                    reachable.append(result)

        return sorted(reachable)

    def establish(self, device_id: str) -> AgentSession:
        """Return an AgentSession for the given host."""
        return AgentSession(device_id, self._port, self._token)


def _local_subnet() -> str | None:
    """Best-effort detection of the local /24 subnet (e.g. '192.168.1.0/24')."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        return ".".join(parts[:3]) + ".0/24"
    except Exception:
        return None

"""
Progenitor Agent — lightweight daemon that runs on the target device.

The customer installs and starts this once. After that, any operator on the
same network with the correct token can reach the device with no further
action from the customer.

HOW TO INSTALL ON TARGET DEVICE
---------------------------------
The target device only needs Python 3.9+ (already on most machines).

  # Option A: if progenitor is installed on target
  progenitor agent start --token <token>

  # Option B: run as a standalone script (no install needed)
  python3 progenitor_agent.py --token <token>

  # Option C: one-liner (Linux/Mac)
  pip install progenitor && progenitor agent start --token <token>

SECURITY
---------
- Every command must include the correct token (shared secret).
  Requests with wrong/missing token are rejected immediately.
- By default the agent binds to 0.0.0.0 (all interfaces) so it is reachable
  over LAN. Use --host 127.0.0.1 to restrict to localhost only.
- The agent does NOT require root. Levers that need root (e.g. CPU governor)
  will fail gracefully on the device side if the user running the agent lacks
  the required privilege.
- Stop the agent any time: Ctrl+C or kill the process.

PROTOCOL
---------
Simple TCP, newline-delimited JSON.

  Request  (operator → agent):
      {"token": "<tok>", "cmd": "<shell command>"}\n

  Response (agent → operator):
      {"success": true/false, "stdout": "...", "stderr": "...",
       "returncode": 0, "error": null}\n
"""

from __future__ import annotations

import json
import os
import secrets
import socketserver
import subprocess
import sys
import threading


DEFAULT_PORT = 7777
_TOKEN: str = ""   # set by start_server() before accepting connections


# ---------------------------------------------------------------------------
# TCP request handler
# ---------------------------------------------------------------------------

class _AgentHandler(socketserver.BaseRequestHandler):
    """Handle one incoming connection from the operator."""

    def handle(self) -> None:
        try:
            raw = self._recv_line()
            if not raw:
                return
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                self._send({"success": False, "error": "invalid JSON"})
                return

            # Authentication
            if req.get("token") != _TOKEN:
                self._send({"success": False, "error": "invalid token"})
                return

            cmd = req.get("cmd", "").strip()
            if not cmd:
                self._send({"success": False, "error": "empty command"})
                return

            # Execute the command on this device
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self._send({
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "error": None,
                })
            except subprocess.TimeoutExpired:
                self._send({"success": False, "error": "command timed out (120s)"})
            except Exception as e:
                self._send({"success": False, "error": str(e)})

        except Exception as e:
            try:
                self._send({"success": False, "error": f"handler error: {e}"})
            except Exception:
                pass

    def _recv_line(self) -> str:
        """Read bytes until newline (handles TCP fragmentation)."""
        data = b""
        while True:
            chunk = self.request.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        line = data.split(b"\n")[0]
        return line.decode(errors="replace")

    def _send(self, obj: dict) -> None:
        self.request.sendall(json.dumps(obj).encode() + b"\n")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def generate_token() -> str:
    """Generate a cryptographically random token."""
    return secrets.token_hex(16)  # 32-char hex string


def start_server(
    token: str,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    *,
    background: bool = False,
) -> _ThreadedTCPServer:
    """
    Start the agent server. Blocks until Ctrl+C unless background=True.

    Args:
        token:      Shared secret. Operators must include this in every request.
        host:       Bind address. 0.0.0.0 = all interfaces (default).
        port:       Listen port (default 7777).
        background: If True, start in a daemon thread and return immediately.
                    If False (default), block until interrupted.
    """
    global _TOKEN
    _TOKEN = token

    server = _ThreadedTCPServer((host, port), _AgentHandler)

    if background:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

    try:
        print(f"Progenitor Agent listening on {host}:{port}")
        print(f"Token: {token}")
        print("Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAgent stopped.")
        server.shutdown()
    return server


# ---------------------------------------------------------------------------
# Standalone entry point (python progenitor_agent.py --token <tok>)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="progenitor-agent",
        description="Progenitor Agent: run on the target device so the operator can enhance it.",
    )
    parser.add_argument("--token", default=None,
                        help="Shared secret token. If omitted a random one is generated and printed.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default {DEFAULT_PORT}).")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default 0.0.0.0 = all interfaces).")
    args = parser.parse_args()

    token = args.token or generate_token()
    if not args.token:
        print(f"Generated token: {token}")
        print("Give this token to the operator — they will need it to connect.")
        print()

    start_server(token, host=args.host, port=args.port, background=False)


if __name__ == "__main__":
    _main()

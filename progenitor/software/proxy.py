"""Local HTTP proxy: cache + gzip so any site is faster through localhost."""

from __future__ import annotations

import gzip
import io
import socket
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from progenitor.software.measure import SiteMeasurement
from urllib.error import URLError

import ssl

# Reuse SSL helper from measure
def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _is_ssl_failure(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError):
        return True
    return False


class _ProxyHandler(BaseHTTPRequestHandler):
    """Single-origin proxy: GET to localhost path -> fetch origin_base + path, cache, optionally gzip."""

    origin_base: str = ""
    cache: dict = {}
    compress: bool = True

    def log_message(self, format: str, *args) -> None:
        pass  # Quiet by default

    def do_HEAD(self) -> None:
        # Support HEAD for health checks and curl -I (same as GET but no body)
        path = self.path.split("?")[0] or "/"
        cache_key = path
        if cache_key in self.cache:
            body, headers = self.cache[cache_key]
            self.send_response(200)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        url = self.origin_base.rstrip("/") + path
        if "?" in self.path:
            url += "?" + self.path.split("?", 1)[1]
        try:
            ctx = _ssl_ctx()
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "Progenitor-Proxy/1.0")
            with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() != "transfer-encoding":
                        self.send_header(k, v)
                self.end_headers()
        except Exception as e:
            if _is_ssl_failure(e):
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "Progenitor-Proxy/1.0")
                with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                    self.send_response(r.status)
                    for k, v in r.headers.items():
                        if k.lower() != "transfer-encoding":
                            self.send_header(k, v)
                    self.end_headers()
            else:
                self.send_error(502, str(e))

    def do_GET(self) -> None:
        path = self.path.split("?")[0] or "/"
        cache_key = path
        if cache_key in self.cache:
            body, headers = self.cache[cache_key]
            self.send_response(200)
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            return

        url = self.origin_base.rstrip("/") + path
        if "?" in self.path:
            url += "?" + self.path.split("?", 1)[1]
        try:
            ctx = _ssl_ctx()
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "Progenitor-Proxy/1.0")
            with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                body = r.read()
                status = r.status
                headers_list = [(k, v) for k, v in r.headers.items()]
        except Exception as e:
            if _is_ssl_failure(e):
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(url, method="GET")
                req.add_header("User-Agent", "Progenitor-Proxy/1.0")
                with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                    body = r.read()
                    status = r.status
                    headers_list = [(k, v) for k, v in r.headers.items()]
            else:
                self.send_error(502, str(e))
                return

        # Optionally compress if not already
        out_headers: list[tuple[str, str]] = []
        skip_headers = {"content-encoding", "content-length", "transfer-encoding"}
        for k, v in headers_list:
            if k.lower() not in skip_headers:
                out_headers.append((k, v))

        if self.compress and len(body) > 512:
            ce = next((v for k, v in headers_list if k.lower() == "content-encoding"), "")
            if not ce or "gzip" not in ce.lower():
                body = gzip.compress(body, compresslevel=6)
                out_headers.append(("Content-Encoding", "gzip"))
                out_headers.append(("Content-Length", str(len(body))))

        # Cache-friendly: add short max-age if missing
        has_cc = any(h[0].lower() == "cache-control" for h in out_headers)
        if not has_cc:
            out_headers.append(("Cache-Control", "max-age=60"))

        self.cache[cache_key] = (body, out_headers)

        self.send_response(status)
        for k, v in out_headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ProgenitorProxy:
    """Local proxy server. Start with start(), stop with stop()."""

    def __init__(self, origin: str, port: Optional[int] = None, compress: bool = True) -> None:
        self.origin = origin.rstrip("/") or origin
        if not self.origin.startswith("http"):
            self.origin = "https://" + self.origin
        self.port = port or _find_free_port()
        self.compress = compress
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        cache: dict = {}
        handler = type("_Handler", (_ProxyHandler,), {
            "origin_base": self.origin,
            "cache": cache,
            "compress": self.compress,
        })

        self._server = HTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def serve_standalone(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    origin: Optional[str] = None,
) -> None:
    """
    Run the proxy as a standalone service (e.g. in Docker or Cloud Run).
    Reads ORIGIN and PORT from environment if not passed. Binds to host (default 0.0.0.0)
    so the service is reachable from outside. Blocks until shutdown.
    """
    import os
    o = origin or os.environ.get("ORIGIN", "").strip()
    if not o:
        raise SystemExit("Set ORIGIN to your site URL (e.g. https://example.com). No code changes needed.")
    if not o.startswith("http://") and not o.startswith("https://"):
        o = "https://" + o
    p = port
    if p is None:
        try:
            p = int(os.environ.get("PORT", "8080"))
        except ValueError:
            p = 8080
    cache: dict = {}
    handler = type("_StandaloneHandler", (_ProxyHandler,), {
        "origin_base": o.rstrip("/"),
        "cache": cache,
        "compress": True,
    })
    server = HTTPServer((host, p), handler)
    print(f"Progenitor proxy serving at http://{host}:{p} -> {o}")
    print("Point your domain here. Same URL, faster for everyone. Ctrl+C to stop.")
    server.serve_forever()


def run_proxy_and_measure(
    url: str,
    warmup: int = 3,
    repeat: int = 20,
    timeout: int = 15,
) -> tuple[SiteMeasurement, SiteMeasurement, float, ProgenitorProxy]:
    """
    Measure direct (before), start proxy, measure through proxy (after).
    Returns (before, after, speedup, proxy). Proxy is left running; caller keeps it or exits.
    """
    from progenitor.software.measure import measure

    before = measure(url, warmup=warmup, repeat=repeat, timeout=timeout)
    proxy = ProgenitorProxy(origin=url, compress=True)
    proxy.start()
    base = proxy.base_url()
    after = measure(base, warmup=warmup, repeat=repeat, timeout=timeout)
    speedup = before.total_ms / after.total_ms if after.total_ms > 0 else 0.0
    return before, after, speedup, proxy

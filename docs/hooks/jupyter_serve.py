"""MkDocs hook: local Jupyter launcher for live-code (dev only).

Starts a small helper on 127.0.0.1:8889 during ``mkdocs serve`` so the browser can
spawn ``scripts/docs-jupyter.sh`` without a full MkDocs restart. Also wraps the
MkDocs dev server when ``on_serve`` runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("mkdocs.plugins.jupyter_serve")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JUPYTER_SCRIPT = _REPO_ROOT / "scripts" / "docs-jupyter.sh"
_JUPYTER_BASE = os.environ.get("RETARGET_JUPYTER_URL", "http://127.0.0.1:8888/").rstrip("/")
_JUPYTER_TOKEN = os.environ.get("JUPYTER_TOKEN", "retarget-docs")
_LAUNCHER_PORT = int(os.environ.get("RETARGET_LAUNCHER_PORT", "8889"))
_LAUNCHER_PREFIX = "/__retarget/jupyter"

_process: subprocess.Popen[bytes] | None = None
_lock = threading.Lock()
_serve_mode = False
_helper_httpd: ThreadingHTTPServer | None = None
_helper_thread: threading.Thread | None = None


def _client_is_local(environ: dict[str, Any]) -> bool:
    addr = environ.get("REMOTE_ADDR") or ""
    return addr in {"127.0.0.1", "::1"} or addr.startswith("127.")


def _jupyter_running() -> bool:
    url = f"{_JUPYTER_BASE}/api?token={_JUPYTER_TOKEN}"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _spawn_jupyter() -> tuple[bool, str]:
    global _process

    if not _JUPYTER_SCRIPT.is_file():
        return False, "missing_script"

    with _lock:
        if _jupyter_running():
            return True, "running"

        if _process is not None and _process.poll() is None:
            return True, "starting"

        try:
            _process = subprocess.Popen(
                ["bash", str(_JUPYTER_SCRIPT)],
                cwd=str(_REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            log.error("Failed to start docs Jupyter: %s", exc)
            return False, "spawn_failed"

        return True, "spawned"


def _cors_headers(environ: dict[str, Any]) -> list[tuple[str, str]]:
    origin = environ.get("HTTP_ORIGIN")
    if origin and ("localhost" in origin or "127.0.0.1" in origin):
        allow_origin = origin
    else:
        allow_origin = "*"
    return [
        ("Access-Control-Allow-Origin", allow_origin),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]


def _json_response(
    start_response: Any, status: str, payload: dict[str, Any], environ: dict[str, Any]
) -> list[bytes]:
    body = json.dumps(payload).encode()
    headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    headers.extend(_cors_headers(environ))
    start_response(status, headers)
    return [body]


def _handle_launcher_request(
    environ: dict[str, Any], start_response: Any
) -> list[bytes] | None:
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    if not path.startswith(_LAUNCHER_PREFIX):
        return None

    if not _client_is_local(environ):
        return _json_response(
            start_response,
            "403 Forbidden",
            {"ok": False, "error": "local_only"},
            environ,
        )

    if method == "OPTIONS":
        start_response("204 No Content", _cors_headers(environ))
        return [b""]

    if path == f"{_LAUNCHER_PREFIX}/status" and method == "GET":
        return _json_response(
            start_response,
            "200 OK",
            {
                "launcher": True,
                "running": _jupyter_running(),
                "script": str(_JUPYTER_SCRIPT),
            },
            environ,
        )

    if path == f"{_LAUNCHER_PREFIX}/start" and method == "POST":
        if _jupyter_running():
            return _json_response(
                start_response,
                "200 OK",
                {"ok": True, "state": "running"},
                environ,
            )

        ok, state = _spawn_jupyter()
        if not ok:
            return _json_response(
                start_response,
                "503 Service Unavailable",
                {"ok": False, "state": state},
                environ,
            )

        return _json_response(
            start_response,
            "200 OK",
            {"ok": True, "state": state},
            environ,
        )

    return _json_response(
        start_response,
        "404 Not Found",
        {"ok": False, "error": "not_found"},
        environ,
    )


def _wrap_mkdocs_server(server: Any) -> Any:
    original = server.serve_request

    def serve_request(environ: dict[str, Any], start_response: Any) -> Any:
        handled = _handle_launcher_request(environ, start_response)
        if handled is not None:
            return handled
        return original(environ, start_response)

    server.set_app(serve_request)
    return server


class _LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        log.debug(format, *args)

    def _dispatch(self) -> None:
        environ = {
            "REQUEST_METHOD": self.command,
            "PATH_INFO": urllib.parse.urlparse(self.path).path,
            "REMOTE_ADDR": self.client_address[0],
            "HTTP_ORIGIN": self.headers.get("Origin", ""),
        }

        chunks: list[bytes] = []

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            self.send_response(int(status.split()[0]))
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()

        handled = _handle_launcher_request(environ, start_response)
        if handled is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"not_found"}')
            return

        for chunk in handled:
            self.wfile.write(chunk)

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:
        self._dispatch()


def _ensure_helper_server() -> None:
    global _helper_httpd, _helper_thread

    if _helper_httpd is not None:
        return

    try:
        _helper_httpd = ThreadingHTTPServer(("127.0.0.1", _LAUNCHER_PORT), _LauncherHandler)
    except OSError as exc:
        log.warning("Could not start Jupyter launcher on port %s: %s", _LAUNCHER_PORT, exc)
        return

    _helper_thread = threading.Thread(target=_helper_httpd.serve_forever, daemon=True)
    _helper_thread.start()
    log.info(
        "Live-code Jupyter launcher at http://127.0.0.1:%s%s/start",
        _LAUNCHER_PORT,
        _LAUNCHER_PREFIX,
    )


def _stop_helper_server() -> None:
    global _helper_httpd, _helper_thread

    if _helper_httpd is None:
        return

    _helper_httpd.shutdown()
    _helper_httpd.server_close()
    _helper_httpd = None
    if _helper_thread is not None:
        _helper_thread.join(timeout=2)
        _helper_thread = None


def on_startup(*, command: str, dirty: bool) -> None:
    global _serve_mode

    _serve_mode = command == "serve"
    if _serve_mode:
        _ensure_helper_server()


def on_post_build(*, config: Any, **kwargs: Any) -> None:
    if _serve_mode:
        _ensure_helper_server()


def on_shutdown() -> None:
    _stop_helper_server()


def on_serve(server: Any, /, *, config: Any, builder: Any) -> Any:
    _ensure_helper_server()
    _wrap_mkdocs_server(server)
    log.info("Live-code Jupyter launcher also on MkDocs dev server at %s/start", _LAUNCHER_PREFIX)
    return server

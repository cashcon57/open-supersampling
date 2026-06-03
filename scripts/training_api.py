#!/usr/bin/env python3
"""Training control API — start/stop/resume OSS training from the dashboard.

Listens ONLY on the Tailscale IP so it's unreachable from the public internet.
Requires a shared secret in the Authorization header.
Uses only Python stdlib — no pip deps needed.
"""

import json
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

LISTEN_HOST = os.environ.get("LISTEN_HOST", "100.126.238.73")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8765"))
SHARED_SECRET = os.environ.get("SHARED_SECRET", "")
COMPOSE_DIR = Path(os.environ.get("COMPOSE_DIR", "/workspace/oss-gaussian/docker/trainer"))
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "oss-trainer")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
}


def check_auth(handler) -> bool:
    auth = handler.headers.get("Authorization", "")
    expected = f"Bearer {SHARED_SECRET}"
    return auth == expected and SHARED_SECRET != ""


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker"] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )


def docker_compose(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_DIR / "docker-compose.yml")] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )


def container_status() -> dict:
    """Return the current state of the oss-trainer container."""
    result = docker("inspect", CONTAINER_NAME)
    if result.returncode != 0:
        return {"running": False, "state": "not-found", "error": result.stderr.strip()}

    try:
        data = json.loads(result.stdout)
        info = data[0]
        state = info.get("State", {})
        return {
            "running": state.get("Running", False),
            "state": state.get("Status", "exited"),
            "exit_code": state.get("ExitCode"),
            "started_at": state.get("StartedAt", ""),
            "finished_at": state.get("FinishedAt", ""),
        }
    except (json.JSONDecodeError, IndexError, KeyError):
        return {"running": False, "state": "unknown", "error": "parse failed"}


def recent_logs(lines: int = 20) -> str:
    result = docker("logs", "--tail", str(lines), CONTAINER_NAME)
    return result.stdout


def json_response(handler, data: dict, status: int = 200):
    body = json.dumps(data, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    for key, value in CORS_HEADERS.items():
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, text: str, status: int = 200):
    body = text.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain")
    for key, value in CORS_HEADERS.items():
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[api] {self.client_address[0]} - {args[0]}", file=sys.stderr)

    def do_OPTIONS(self):
        self.send_response(204)
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

    def do_GET(self):
        # /health is public (Docker HEALTHCHECK, no auth)
        if self.path == "/health":
            json_response(self, {"ok": True})
            return

        if not check_auth(self):
            json_response(self, {"error": "unauthorized"}, 401)
            return

        if self.path == "/status":
            status = container_status()
            status["logs_tail"] = recent_logs()
            json_response(self, status)

        else:
            json_response(self, {"error": "not found"}, 404)

    def do_POST(self):
        if not check_auth(self):
            json_response(self, {"error": "unauthorized"}, 401)
            return

        if self.path == "/start":
            status = container_status()
            if status["running"]:
                json_response(self, {"action": "start", "result": "already-running", "status": status})
                return
            result = docker_compose("up", "-d")
            json_response(self, {
                "action": "start",
                "result": "started" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })

        elif self.path == "/stop":
            result = docker_compose("stop")
            json_response(self, {
                "action": "stop",
                "result": "stopped" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })

        elif self.path == "/restart":
            result = docker_compose("restart")
            json_response(self, {
                "action": "restart",
                "result": "restarted" if result.returncode == 0 else "failed",
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })

        else:
            json_response(self, {"error": "not found"}, 404)


def main():
    if not SHARED_SECRET:
        print("FATAL: SHARED_SECRET env var not set", file=sys.stderr)
        sys.exit(1)

    server = HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"[api] Listening on {LISTEN_HOST}:{LISTEN_PORT}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] Shutting down", file=sys.stderr)
        server.server_close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tailnet-only training control API for OpenSuperSampling.

Security model:
- The server binds only to the host's Tailscale IP.
- No bearer token/auth; Tailscale ACLs are the auth boundary.
- Browser-origin state changes must include X-OSS-Tailnet-Control to avoid blind
  cross-site POSTs from arbitrary pages loaded by a tailnet-connected browser.
- If TLS_CERT_FILE/TLS_KEY_FILE are set, serves HTTPS for mixed-content-safe
  calls from https://opensupersampling.org.
"""

from __future__ import annotations

import json
import os
import ssl
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

LISTEN_HOST = os.environ.get("LISTEN_HOST", "100.126.238.73")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8765"))
COMPOSE_DIR = Path(os.environ.get("COMPOSE_DIR", "/home/cashc/OpenSuperSampling/docker/trainer"))
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "oss-trainer")
COMPOSE_SERVICE = os.environ.get("COMPOSE_SERVICE", "oss-trainer")
PAUSE_STOP_TIMEOUT = int(os.environ.get("PAUSE_STOP_TIMEOUT", "900"))
CHECKPOINT_DIR = Path(os.environ.get(
    "CHECKPOINT_DIR",
    "/home/cashc/checkpoints/srcnn-v7.0-pico-005",
))
TRAINING_PROGRESS_STALE_SECONDS = int(os.environ.get("TRAINING_PROGRESS_STALE_SECONDS", "1800"))
SCHEDULE_FILE = Path(os.environ.get(
    "SCHEDULE_FILE",
    "/home/cashc/OpenSuperSampling/.state/training_resume_schedule.json",
))
TLS_CERT_FILE = os.environ.get("TLS_CERT_FILE", "")
TLS_KEY_FILE = os.environ.get("TLS_KEY_FILE", "")

DEFAULT_ALLOWED_ORIGINS = ",".join([
    "https://opensupersampling.org",
    f"http://{LISTEN_HOST}:{LISTEN_PORT}",
    f"https://{LISTEN_HOST}:{LISTEN_PORT}",
    "http://homelab-desktop.taild5a730.ts.net:8765",
    "https://homelab-desktop.taild5a730.ts.net:8765",
])
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
}
CONTROL_HEADER = "X-OSS-Tailnet-Control"

_schedule_lock = threading.Lock()
_schedule_timer: threading.Timer | None = None


def now_ts() -> float:
    return time.time()


def iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def compose_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HOME", "/home/cashc")
    env.setdefault("OSS_REPO", "/home/cashc/OpenSuperSampling")
    env.setdefault("OSS_DATASETS", "/home/cashc/datasets")
    env.setdefault("OSS_CHECKPOINTS", "/home/cashc/checkpoints")
    return env


def run_cmd(args: list[str], timeout: int = 120, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=compose_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run_cmd(["docker", *args], timeout=timeout)


def docker_compose(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["docker", "compose", "-f", str(COMPOSE_DIR / "docker-compose.yml"), *args],
        timeout=timeout,
        cwd=COMPOSE_DIR,
    )


def completed_to_json(result: subprocess.CompletedProcess[str], ok_result: str, fail_result: str = "failed") -> dict[str, Any]:
    return {
        "result": ok_result if result.returncode == 0 else fail_result,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "returncode": result.returncode,
    }


def trainer_progress_status() -> dict[str, Any]:
    """Return host-side trainer progress freshness.

    Docker's image healthcheck used to watch history.jsonl directly, but history
    is sparse at --log-every cadence. Prefer the per-step heartbeat added to the
    trainer, and fall back to history.jsonl for already-running old containers.
    """
    candidates = [
        CHECKPOINT_DIR / ".trainer_heartbeat.json",
        CHECKPOINT_DIR / "history.jsonl",
    ]
    newest_path: Path | None = None
    newest_mtime = 0.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = path
    if newest_path is None:
        return {"fresh": False, "path": None, "age_seconds": None}
    age = max(0.0, now_ts() - newest_mtime)
    return {
        "fresh": age < TRAINING_PROGRESS_STALE_SECONDS,
        "path": str(newest_path),
        "age_seconds": int(round(age)),
        "stale_after_seconds": TRAINING_PROGRESS_STALE_SECONDS,
    }


def container_status() -> dict[str, Any]:
    result = docker("inspect", CONTAINER_NAME)
    if result.returncode != 0:
        return {"running": False, "paused": False, "state": "not-found", "error": result.stderr.strip()}

    try:
        info = json.loads(result.stdout)[0]
        state = info.get("State", {})
        docker_health = state.get("Health", {}).get("Status")
        progress = trainer_progress_status()
        effective_health = docker_health
        health_note = None
        if state.get("Running", False) and docker_health == "unhealthy" and progress.get("fresh"):
            effective_health = "healthy"
            health_note = "docker healthcheck is stale; trainer progress file is fresh"
        return {
            "running": bool(state.get("Running", False)),
            "paused": bool(state.get("Paused", False)),
            "restarting": bool(state.get("Restarting", False)),
            "state": state.get("Status", "exited"),
            "health": effective_health,
            "docker_health": docker_health,
            "health_note": health_note,
            "progress": progress,
            "exit_code": state.get("ExitCode"),
            "started_at": state.get("StartedAt", ""),
            "finished_at": state.get("FinishedAt", ""),
        }
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        return {"running": False, "paused": False, "state": "unknown", "error": f"parse failed: {exc}"}


def recent_logs(lines: int = 30) -> str:
    result = docker("logs", "--tail", str(lines), CONTAINER_NAME)
    return (result.stdout + result.stderr)[-4000:]


def pause_training() -> tuple[dict[str, Any], subprocess.CompletedProcess[str] | None]:
    status = container_status()
    if not status.get("running"):
        return {"result": "already-paused", "status": status}, None
    if status.get("paused"):
        docker("unpause", CONTAINER_NAME, timeout=30)
    result = docker_compose(
        "stop", "-t", str(PAUSE_STOP_TIMEOUT), COMPOSE_SERVICE,
        timeout=PAUSE_STOP_TIMEOUT + 30,
    )
    return {**completed_to_json(result, "paused-and-unloaded"), "status": container_status()}, result


def resume_training() -> tuple[dict[str, Any], subprocess.CompletedProcess[str] | None]:
    status = container_status()
    if status.get("running") and status.get("paused"):
        result = docker("unpause", CONTAINER_NAME, timeout=30)
        return {**completed_to_json(result, "resumed"), "status": container_status()}, result
    if status.get("running"):
        return {"result": "already-running", "status": status}, None
    result = docker_compose("up", "-d", COMPOSE_SERVICE)
    return {**completed_to_json(result, "resumed"), "status": container_status()}, result


def _read_schedule_unlocked() -> dict[str, Any] | None:
    if not SCHEDULE_FILE.exists():
        return None
    try:
        data = json.loads(SCHEDULE_FILE.read_text())
        resume_at = float(data["resume_at"])
        return {**data, "resume_at": resume_at}
    except Exception as exc:
        return {"error": f"schedule parse failed: {exc}", "path": str(SCHEDULE_FILE)}


def scheduled_resume_status() -> dict[str, Any] | None:
    with _schedule_lock:
        data = _read_schedule_unlocked()
    if not data or data.get("error"):
        return data
    remaining = max(0, float(data["resume_at"]) - now_ts())
    return {
        **data,
        "resume_at_iso": iso_from_ts(float(data["resume_at"])),
        "remaining_seconds": int(round(remaining)),
    }


def _write_schedule_unlocked(data: dict[str, Any]) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCHEDULE_FILE.with_suffix(SCHEDULE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, SCHEDULE_FILE)


def _clear_schedule_unlocked() -> None:
    global _schedule_timer
    if _schedule_timer:
        _schedule_timer.cancel()
        _schedule_timer = None
    SCHEDULE_FILE.unlink(missing_ok=True)


def cancel_scheduled_resume() -> dict[str, Any]:
    with _schedule_lock:
        existed = SCHEDULE_FILE.exists()
        _clear_schedule_unlocked()
    return {"result": "cancelled" if existed else "none-scheduled", "scheduled_resume": None}


def _auto_resume_callback() -> None:
    print("[api] scheduled resume fired", file=sys.stderr)
    payload, result = resume_training()
    ok = result is None or result.returncode == 0
    with _schedule_lock:
        if ok:
            _clear_schedule_unlocked()
        else:
            # Retry every minute while the persisted schedule exists.
            _arm_schedule_timer_unlocked(60)
    print(f"[api] scheduled resume result: {payload.get('result')}", file=sys.stderr)


def _arm_schedule_timer_unlocked(delay_seconds: float) -> None:
    global _schedule_timer
    if _schedule_timer:
        _schedule_timer.cancel()
    _schedule_timer = threading.Timer(max(0.0, delay_seconds), _auto_resume_callback)
    _schedule_timer.daemon = True
    _schedule_timer.start()


def schedule_resume_after(seconds: int, source: str = "control-page") -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("duration must be greater than zero")
    if seconds > 7 * 24 * 3600:
        raise ValueError("duration cannot exceed 7 days")
    resume_at = now_ts() + seconds
    data = {
        "resume_at": resume_at,
        "resume_at_iso": iso_from_ts(resume_at),
        "duration_seconds": seconds,
        "created_at": now_ts(),
        "created_at_iso": iso_from_ts(now_ts()),
        "source": source,
    }
    with _schedule_lock:
        _write_schedule_unlocked(data)
        _arm_schedule_timer_unlocked(seconds)
    return scheduled_resume_status() or data


def restore_schedule_from_disk() -> None:
    with _schedule_lock:
        data = _read_schedule_unlocked()
        if not data or data.get("error"):
            return
        delay = float(data["resume_at"]) - now_ts()
        _arm_schedule_timer_unlocked(delay)
        print(f"[api] restored scheduled resume in {max(0, int(delay))}s", file=sys.stderr)


def control_page() -> bytes:
    timeout_ms = (PAUSE_STOP_TIMEOUT + 60) * 1000
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OSS Tailnet Training Controls</title>
  <style>
    body{{background:#06110f;color:#e9fff7;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:24px;max-width:980px}}
    h1{{font-size:20px;color:#8cffc1}}.sub{{color:#a8d4a0;font-size:13px}}.row{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}
    button,input{{background:#071611;color:#8cffc1;border:1px solid rgba(140,255,193,.68);padding:10px 14px;font:inherit}}
    input{{width:72px;color:#e9fff7}}button{{cursor:pointer}}button:hover{{background:#0b241b}}.danger{{color:#fca5a5;border-color:#b91c1c}}.warn{{color:#fbbf24;border-color:#b45309}}
    pre{{white-space:pre-wrap;background:#020807;border:1px solid rgba(140,255,193,.26);padding:12px;min-height:120px;max-height:55vh;overflow:auto}}
    .badge{{display:inline-block;border:1px solid rgba(140,255,193,.35);padding:2px 8px;margin-left:8px;color:#a8d4a0}}
    .box{{border:1px solid rgba(140,255,193,.22);padding:12px;margin:12px 0;background:#020807}}
    label{{color:#a8d4a0}}
  </style>
</head>
<body>
  <h1>OpenSuperSampling Training Controls <span id="badge" class="badge">checking</span></h1>
  <p class="sub">Tailnet-only API. If this page loaded, you reached the Tailscale-bound service.</p>
  <div class="row">
    <button onclick="call('start')">Start training</button>
    <button class="warn" onclick="call('pause')">Pause + unload GPU</button>
    <button onclick="call('unpause')">Resume now</button>
    <button class="danger" onclick="call('stop')">Stop</button>
    <button onclick="status()">Refresh status</button>
  </div>
  <div class="box">
    <label>Pause + unload GPU for </label>
    <input id="hours" type="number" min="0" max="168" value="1"> <label>hrs</label>
    <input id="minutes" type="number" min="0" max="59" value="0"> <label>mins</label>
    <button class="warn" onclick="pauseFor()">Pause for duration</button>
    <button onclick="call('cancel_resume')">Cancel scheduled resume</button>
  </div>
  <pre id="out">Loading...</pre>
<script>
const out = document.getElementById('out');
const badge = document.getElementById('badge');
function render(data) {{
  out.textContent = JSON.stringify(data, null, 2);
  const st = data.status || data;
  const sched = data.scheduled_resume || st.scheduled_resume;
  const schedText = sched ? `, resumes in ${{sched.remaining_seconds}}s` : '';
  badge.textContent = (st.paused ? 'paused' : (st.state || data.result || 'ok')) + schedText;
}}
async function status() {{
  try {{
    const r = await fetch('/status', {{ signal: AbortSignal.timeout(10000) }});
    render(await r.json());
  }} catch (e) {{ out.textContent = 'status failed: ' + e.message; badge.textContent='error'; }}
}}
async function call(action) {{
  out.textContent = action + '...';
  try {{
    const timeoutMs = action === 'pause' ? {timeout_ms} : 120000;
    const r = await fetch('/' + action, {{ method:'POST', headers:{{'X-OSS-Tailnet-Control':'1'}}, signal: AbortSignal.timeout(timeoutMs) }});
    render(await r.json());
  }} catch (e) {{ out.textContent = action + ' failed: ' + e.message; badge.textContent='error'; }}
}}
async function pauseFor() {{
  const hours = parseInt(document.getElementById('hours').value || '0', 10);
  const minutes = parseInt(document.getElementById('minutes').value || '0', 10);
  out.textContent = `pause_for ${{hours}}h ${{minutes}}m...`;
  try {{
    const r = await fetch('/pause_for', {{
      method:'POST',
      headers:{{'X-OSS-Tailnet-Control':'1', 'Content-Type':'application/json'}},
      body: JSON.stringify({{hours, minutes}}),
      signal: AbortSignal.timeout({timeout_ms})
    }});
    render(await r.json());
  }} catch (e) {{ out.textContent = 'pause_for failed: ' + e.message; badge.textContent='error'; }}
}}
status();
setInterval(status, 30000);
</script>
</body>
</html>""".encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[api] {self.client_address[0]} - {format % args}", file=sys.stderr)

    @property
    def origin(self) -> str:
        return self.headers.get("Origin", "").rstrip("/")

    @property
    def parsed_path(self):
        return urlparse(self.path)

    def origin_allowed(self) -> bool:
        return not self.origin or self.origin in ALLOWED_ORIGINS

    def send_cors_headers(self) -> None:
        if self.origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", self.origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", f"{CONTROL_HEADER}, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")

    def json_response(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def html_response(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if not self.origin_allowed():
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if not self.origin_allowed():
            self.json_response({"error": "origin not allowed"}, 403)
            return
        path = self.parsed_path.path
        if path in {"/", "/control"}:
            self.html_response(control_page())
            return
        if path == "/health":
            self.json_response({"ok": True, "tls": bool(TLS_CERT_FILE and TLS_KEY_FILE)})
            return
        if path == "/status":
            status = container_status()
            status["logs_tail"] = recent_logs()
            status["scheduled_resume"] = scheduled_resume_status()
            self.json_response(status)
            return
        if path == "/scheduled_resume":
            self.json_response({"scheduled_resume": scheduled_resume_status()})
            return
        self.json_response({"error": "not found"}, 404)

    def require_control_header(self) -> bool:
        if not self.origin_allowed():
            self.json_response({"error": "origin not allowed"}, 403)
            return False
        if self.headers.get(CONTROL_HEADER) != "1":
            self.json_response({"error": f"missing {CONTROL_HEADER}"}, 400)
            return False
        return True

    def read_request_data(self) -> dict[str, Any]:
        query = {k: v[-1] for k, v in parse_qs(self.parsed_path.query).items() if v}
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return query
        raw = self.rfile.read(length).decode(errors="replace")
        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
        else:
            body = {k: v[-1] for k, v in parse_qs(raw).items() if v}
        return {**query, **body}

    def parse_pause_duration(self) -> int:
        data = self.read_request_data()
        seconds = int(float(data.get("seconds", 0) or 0))
        minutes = int(float(data.get("minutes", 0) or 0))
        hours = int(float(data.get("hours", 0) or 0))
        total = seconds + minutes * 60 + hours * 3600
        if total <= 0:
            raise ValueError("provide hours, minutes, or seconds > 0")
        return total

    def do_POST(self) -> None:
        if not self.require_control_header():
            return
        action = self.parsed_path.path.strip("/")
        try:
            if action == "start":
                cancel_scheduled_resume()
                status = container_status()
                if status.get("running"):
                    self.json_response({"action": action, "result": "already-running", "status": status, "scheduled_resume": None})
                    return
                result = docker_compose("up", "-d")
                payload = {"action": action, **completed_to_json(result, "started")}
                payload["status"] = container_status()
                payload["scheduled_resume"] = None
                self.json_response(payload)
                return

            if action == "stop":
                cancel_scheduled_resume()
                status = container_status()
                if status.get("paused"):
                    docker("unpause", CONTAINER_NAME, timeout=30)
                result = docker_compose("stop")
                payload = {"action": action, **completed_to_json(result, "stopped")}
                payload["status"] = container_status()
                payload["scheduled_resume"] = None
                self.json_response(payload)
                return

            if action == "pause":
                cancel_scheduled_resume()
                payload, _ = pause_training()
                payload["action"] = action
                payload["scheduled_resume"] = None
                self.json_response(payload)
                return

            if action == "pause_for":
                seconds = self.parse_pause_duration()
                cancel_scheduled_resume()
                payload, result = pause_training()
                payload["action"] = action
                if result is None or result.returncode == 0:
                    payload["scheduled_resume"] = schedule_resume_after(seconds)
                else:
                    payload["scheduled_resume"] = None
                self.json_response(payload)
                return

            if action in {"unpause", "resume"}:
                cancel_scheduled_resume()
                payload, _ = resume_training()
                payload["action"] = action
                payload["scheduled_resume"] = None
                self.json_response(payload)
                return

            if action == "cancel_resume":
                payload = cancel_scheduled_resume()
                payload["action"] = action
                payload["status"] = container_status()
                self.json_response(payload)
                return

            self.json_response({"error": "not found"}, 404)
        except ValueError as exc:
            self.json_response({"action": action, "result": "bad-request", "stderr": str(exc)}, 400)
        except subprocess.TimeoutExpired as exc:
            self.json_response({"action": action, "result": "timeout", "stderr": str(exc)}, 504)
        except Exception as exc:  # keep API responses JSON even for unexpected Docker errors
            self.json_response({"action": action, "result": "error", "stderr": repr(exc)}, 500)


def main() -> None:
    restore_schedule_from_disk()
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    scheme = "http"
    if TLS_CERT_FILE and TLS_KEY_FILE:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(TLS_CERT_FILE, TLS_KEY_FILE)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"[api] Listening on {scheme}://{LISTEN_HOST}:{LISTEN_PORT}", file=sys.stderr)
    print(f"[api] Compose dir: {COMPOSE_DIR}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()

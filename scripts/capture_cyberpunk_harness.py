#!/usr/bin/env python3
"""Cyberpunk capture readiness harness.

This script is intentionally operator-safe: it does not install DLLs, edit
game files, kill processes, or embed credentials. It can verify/start a local
ingest server, generate Cyberpunk installer input config, optionally launch the
game benchmark when an executable is known, run the existing synthetic capture
simulator, and emit a single JSON readiness report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_capture_installer import (  # noqa: E402
    SCHEDULED_TASK_XML,
    build_config,
    build_installer_manifest,
    normalize_capture_api_base,
    render_wix_source,
    run_fake_cyberpunk_installer_smoke,
)


GAME_ID = "cyberpunk-2077"
GAME_EXE_NAME = "Cyberpunk2077.exe"
DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_BENCHMARK_ARGS = ("--launcher-skip", "-benchmark")
KNOWN_3080TI_CYBERPUNK_EXE = r"D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"
DEFAULT_CREDENTIALS_ENV = REPO_ROOT / ".secrets" / "r2-credentials.env"
DIRECT_R2_REQUIRED = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
WORKER_UPLOAD_REQUIRED = ("WORKER_UPLOAD_URL", "WORKER_SHARED_SECRET")


@dataclass
class StepResult:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = {"name": self.name, "status": self.status}
        if self.details:
            data["details"] = self.details
        return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_token(token: str | None) -> dict[str, Any]:
    if not token:
        return {"present": False}
    return {
        "present": True,
        "length": len(token),
        "preview": f"...{token[-4:]}" if len(token) >= 4 else "...",
    }


def _redact_url(url: str | None) -> dict[str, Any]:
    if not url:
        return {"present": False}
    parsed = urlparse(url)
    return {
        "present": True,
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "path": parsed.path or "/",
    }


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without evaluating shell code."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def build_runtime_env(
    *,
    base_env: Mapping[str, str] | None = None,
    credentials_env: Path = DEFAULT_CREDENTIALS_ENV,
) -> dict[str, str]:
    """Merge process env with gitignored credentials; process env wins."""
    env = dict(os.environ if base_env is None else base_env)
    file_env = load_env_file(credentials_env)
    for key, value in file_env.items():
        env.setdefault(key, value)
    if env.get("CF_ACCOUNT_ID") and not env.get("R2_ENDPOINT"):
        env["R2_ENDPOINT"] = f"https://{env['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return env


def _missing(env: Mapping[str, str], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if not env.get(key)]


def credential_preflight(
    *,
    env: Mapping[str, str],
    credentials_env: Path = DEFAULT_CREDENTIALS_ENV,
) -> StepResult:
    """Report which R2-compatible credential path is available."""
    direct_missing = _missing(env, (*DIRECT_R2_REQUIRED, "R2_ENDPOINT"))
    worker_missing = _missing(env, WORKER_UPLOAD_REQUIRED)
    direct_ready = not direct_missing
    worker_ready = not worker_missing
    if direct_ready and worker_ready:
        status = "both"
    elif direct_ready:
        status = "direct_r2"
    elif worker_ready:
        status = "worker_upload"
    else:
        status = "missing"
    return StepResult(
        "credential_preflight",
        status,
        {
            "credentials_env": str(credentials_env),
            "direct_r2": {
                "ready": direct_ready,
                "missing": direct_missing,
                "bucket": env.get("R2_BUCKET") or "<default>",
                "endpoint": _redact_url(env.get("R2_ENDPOINT")),
                "access_key_id": _redact_token(env.get("R2_ACCESS_KEY_ID")),
                "secret_access_key": {"present": bool(env.get("R2_SECRET_ACCESS_KEY"))},
            },
            "worker_upload": {
                "ready": worker_ready,
                "missing": worker_missing,
                "upload_url": _redact_url(env.get("WORKER_UPLOAD_URL")),
                "shared_secret": {"present": bool(env.get("WORKER_SHARED_SECRET"))},
            },
        },
    )


def worker_upload_url(base_url: str, key: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"WORKER_UPLOAD_URL must be an absolute URL: {base_url!r}")
    base_path = parsed.path.rstrip("/")
    upload_path = f"{base_path}/upload/{key}" if base_path else f"/upload/{key}"
    return urlunparse(parsed._replace(path=upload_path, params="", query="", fragment=""))


def verify_worker_upload_credentials(
    *,
    env: Mapping[str, str],
    timeout: float,
    smoke_key: str | None = None,
) -> StepResult:
    missing = _missing(env, WORKER_UPLOAD_REQUIRED)
    if missing:
        return StepResult(
            "worker_upload_credentials",
            "skipped",
            {"reason": "Worker upload credentials missing", "missing": missing},
        )
    smoke_key = smoke_key or f"test/capture-ingest-smoke/{uuid.uuid4().hex}.json"
    url = worker_upload_url(env["WORKER_UPLOAD_URL"], smoke_key)
    body = json.dumps(
        {"kind": "capture-ingest-smoke", "generated_at": _utc_now()},
        sort_keys=True,
    ).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {env['WORKER_SHARED_SECRET']}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read(4096)
            duration_ms = (time.perf_counter() - started) * 1000.0
            decoded: Any = None
            try:
                decoded = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = response_body.decode("utf-8", errors="replace")
            return StepResult(
                "worker_upload_credentials",
                "ready" if 200 <= int(resp.status) < 300 else "error",
                {
                    "upload_url": _redact_url(env.get("WORKER_UPLOAD_URL")),
                    "smoke_key": smoke_key,
                    "status_code": int(resp.status),
                    "duration_ms": round(duration_ms, 3),
                    "response": decoded,
                    "shared_secret": {"present": True},
                },
            )
    except error.HTTPError as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        return StepResult(
            "worker_upload_credentials",
            "error",
            {
                "upload_url": _redact_url(env.get("WORKER_UPLOAD_URL")),
                "smoke_key": smoke_key,
                "status_code": exc.code,
                "duration_ms": round(duration_ms, 3),
                "error": exc.reason,
                "shared_secret": {"present": True},
            },
        )
    except (OSError, TimeoutError, error.URLError) as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        return StepResult(
            "worker_upload_credentials",
            "error",
            {
                "upload_url": _redact_url(env.get("WORKER_UPLOAD_URL")),
                "smoke_key": smoke_key,
                "duration_ms": round(duration_ms, 3),
                "error": str(exc),
                "shared_secret": {"present": True},
            },
        )


def health_url(server: str) -> str:
    parsed = urlparse(server)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"server must be an absolute URL: {server!r}")
    return urlunparse(parsed._replace(path="/healthz", params="", query="", fragment=""))


def ingest_url(server: str) -> str:
    parsed = urlparse(server)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"server must be an absolute URL: {server!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/ingest") or path == "/ingest":
        return urlunparse(parsed._replace(params="", query="", fragment=""))
    path = f"{path}/ingest" if path else "/ingest"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def probe_ingest(server: str, *, timeout: float = 2.0) -> StepResult:
    url = health_url(server)
    started = time.perf_counter()
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(4096)
            duration_ms = (time.perf_counter() - started) * 1000.0
            decoded: Any = None
            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = body.decode("utf-8", errors="replace")
            return StepResult(
                "verify_ingest",
                "ready" if 200 <= int(resp.status) < 300 else "error",
                {
                    "url": url,
                    "status_code": int(resp.status),
                    "duration_ms": round(duration_ms, 3),
                    "response": decoded,
                },
            )
    except (OSError, TimeoutError, error.URLError) as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        return StepResult(
            "verify_ingest",
            "unavailable",
            {"url": url, "duration_ms": round(duration_ms, 3), "error": str(exc)},
        )


def _host_port(server: str) -> tuple[str, int]:
    parsed = urlparse(server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"server must be an absolute http(s) URL: {server!r}")
    return parsed.hostname, int(parsed.port or (443 if parsed.scheme == "https" else 80))


def _is_loopback(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"}


def _port_listening(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def mint_local_token(token_store: Path, *, label: str = "cyberpunk-readiness") -> tuple[str, StepResult]:
    token = uuid.uuid4().hex
    env = os.environ.copy()
    env["OSS_CAPTURE_TOKEN_STORE"] = str(token_store)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "server.oss_capture_ingest.main",
            "mint-token",
            "--label",
            label,
            "--token",
            token,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    status = "ready" if proc.returncode == 0 else "error"
    return token, StepResult(
        "mint_local_token",
        status,
        {
            "token": _redact_token(token),
            "token_store": str(token_store),
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        },
    )


def maybe_start_ingest(
    server: str,
    *,
    token_store: Path,
    env: Mapping[str, str],
    start: bool,
    keep_running: bool,
    timeout: float,
) -> tuple[subprocess.Popen[str] | None, StepResult]:
    host, port = _host_port(server)
    if not _is_loopback(host):
        return None, StepResult(
            "start_ingest",
            "skipped",
            {"reason": "server is not loopback; start is only supported for local ingest"},
        )
    if _port_listening(host, port):
        return None, StepResult(
            "start_ingest",
            "skipped",
            {"reason": "port already listening", "host": host, "port": port},
        )
    if not start:
        return None, StepResult(
            "start_ingest",
            "skipped",
            {"reason": "pass --start-ingest to launch a local ingest process"},
        )
    missing_direct_r2 = _missing(env, (*DIRECT_R2_REQUIRED, "R2_ENDPOINT"))
    if missing_direct_r2:
        return None, StepResult(
            "start_ingest",
            "skipped",
            {
                "reason": (
                    "local FastAPI ingest needs direct R2 S3 credentials; "
                    "Worker upload credentials are checked separately"
                ),
                "missing_direct_r2": missing_direct_r2,
            },
        )

    proc_env = dict(env)
    proc_env["OSS_CAPTURE_TOKEN_STORE"] = str(token_store)
    log_path = token_store.with_name("ingest-server.log")
    log_handle = log_path.open("ab")
    proc: subprocess.Popen[str] = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "server.oss_capture_ingest.main",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=proc_env,
        text=True,
        stdout=log_handle,
        stderr=log_handle,
    )
    log_handle.close()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_tail = ""
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            except OSError:
                pass
            return proc, StepResult(
                "start_ingest",
                "error",
                {
                    "pid": proc.pid,
                    "returncode": proc.returncode,
                    "log_path": str(log_path),
                    "log_tail": log_tail,
                },
            )
        probe = probe_ingest(server, timeout=0.5)
        if probe.status == "ready":
            return proc, StepResult(
                "start_ingest",
                "ready",
                {
                    "pid": proc.pid,
                    "host": host,
                    "port": port,
                    "keep_running": keep_running,
                    "log_path": str(log_path),
                },
            )
        time.sleep(0.25)
    return proc, StepResult(
        "start_ingest",
        "error",
        {
            "pid": proc.pid,
            "reason": f"healthz did not pass within {timeout:.1f}s",
            "log_path": str(log_path),
        },
    )


def generate_installer_inputs(
    output_dir: Path,
    *,
    server: str,
    token: str | None,
    mode: str,
    installer_version: str,
) -> StepResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        game_id=GAME_ID,
        game_exe_name=GAME_EXE_NAME,
        proxy_dll_name="dxgi.dll",
        installer_version=installer_version,
        capture_api_base=normalize_capture_api_base(server),
        install_token=token,
        capture_mode=mode,
    )
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "installer_manifest.json"
    scheduled_task_path = output_dir / "scheduled_task.xml"
    wix_path = output_dir / "oss_capture_installer.wxs"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = build_installer_manifest(config, output_dir=str(output_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scheduled_task_path.write_text(SCHEDULED_TASK_XML, encoding="utf-8")
    wix_path.write_text(render_wix_source(manifest), encoding="utf-8")
    return StepResult(
        "generate_installer_config",
        "ready",
        {
            "config_path": str(config_path),
            "manifest_path": str(manifest_path),
            "scheduled_task_path": str(scheduled_task_path),
            "wix_path": str(wix_path),
            "game_id": GAME_ID,
            "mode": mode,
            "install_token": _redact_token(config.get("install_token")),
        },
    )


def smoke_fake_cyberpunk_installer(
    output_dir: Path,
    *,
    keep_workdir: bool,
) -> StepResult:
    manifest_path = output_dir / "installer_manifest.json"
    if not manifest_path.is_file():
        return StepResult(
            "installer_fake_cyberpunk_smoke",
            "error",
            {"reason": "installer manifest has not been generated", "manifest_path": str(manifest_path)},
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = run_fake_cyberpunk_installer_smoke(
        manifest,
        work_dir=output_dir,
        artifact_dir=manifest.get("artifact_dir"),
        keep_workdir=keep_workdir,
    )
    status = result.pop("status", "error")
    return StepResult("installer_fake_cyberpunk_smoke", status, result)


def candidate_cyberpunk_paths(
    *,
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
    system: str | None = None,
) -> list[Path]:
    env = env or os.environ
    paths: list[Path] = []
    for value in (explicit, env.get("CYBERPUNK_EXE"), env.get("CYBERPUNK2077_EXE")):
        if value:
            paths.append(Path(value).expanduser())
    if (system or platform.system()).lower() == "windows":
        paths.append(Path(KNOWN_3080TI_CYBERPUNK_EXE))
        for root in (
            env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            env.get("ProgramFiles", r"C:\Program Files"),
        ):
            paths.extend(
                [
                    Path(root) / "Steam" / "steamapps" / "common" / "Cyberpunk 2077" / "bin" / "x64" / GAME_EXE_NAME,
                    Path(root) / "GOG Galaxy" / "Games" / "Cyberpunk 2077" / "bin" / "x64" / GAME_EXE_NAME,
                    Path(root) / "Epic Games" / "Cyberpunk 2077" / "bin" / "x64" / GAME_EXE_NAME,
                ]
            )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def find_cyberpunk_exe(
    *,
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
    system: str | None = None,
) -> tuple[Path | None, list[str]]:
    candidates = candidate_cyberpunk_paths(explicit=explicit, env=env, system=system)
    for path in candidates:
        if path.is_file():
            return path, [str(candidate) for candidate in candidates]
    return None, [str(candidate) for candidate in candidates]


def maybe_launch_cyberpunk(
    *,
    explicit_exe: str | None,
    launch: bool,
    benchmark_args: Sequence[str],
) -> StepResult:
    exe, candidates = find_cyberpunk_exe(explicit=explicit_exe)
    if exe is None:
        return StepResult(
            "launch_cyberpunk_benchmark",
            "skipped",
            {"reason": "Cyberpunk executable not found", "candidates": candidates},
        )
    if not launch:
        return StepResult(
            "launch_cyberpunk_benchmark",
            "skipped",
            {
                "reason": "pass --launch-cyberpunk to start the benchmark",
                "exe": str(exe),
                "benchmark_args": list(benchmark_args),
            },
        )
    proc = subprocess.Popen(
        [str(exe), *benchmark_args],
        cwd=str(exe.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return StepResult(
        "launch_cyberpunk_benchmark",
        "started",
        {"exe": str(exe), "pid": proc.pid, "benchmark_args": list(benchmark_args)},
    )


def run_synthetic_capture(
    *,
    server: str,
    token: str | None,
    frames: int,
    mode: str,
    timeout: float,
) -> StepResult:
    if not token:
        return StepResult(
            "synthetic_capture",
            "skipped",
            {"reason": "no token supplied; set --token or OSS_CAPTURE_INSTALL_TOKEN"},
        )
    cmd = [
        sys.executable,
        "scripts/capture_simulator.py",
        "simulate-session",
        "--game",
        GAME_ID,
        "--frames",
        str(frames),
        "--mode",
        mode,
        "--server",
        ingest_url(server),
        "--token",
        token,
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    redacted_cmd = list(cmd)
    if "--token" in redacted_cmd:
        token_index = redacted_cmd.index("--token") + 1
        if token_index < len(redacted_cmd):
            redacted_cmd[token_index] = "<redacted-token>"
    details: dict[str, Any] = {
        "returncode": proc.returncode,
        "command": " ".join(shlex.quote(part) for part in redacted_cmd),
    }
    try:
        details["report"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        details["stdout"] = proc.stdout.strip()
    if proc.stderr.strip():
        details["stderr"] = proc.stderr.strip()
    return StepResult(
        "synthetic_capture",
        "ready" if proc.returncode == 0 else "error",
        details,
    )


def build_readiness_report(
    *,
    steps: Sequence[StepResult],
    server: str,
    output_dir: Path,
    token: str | None,
) -> dict[str, Any]:
    statuses = {step.name: step.status for step in steps}
    installer_smoke_ok = statuses.get("installer_fake_cyberpunk_smoke", "skipped") in {"ready", "skipped"}
    local_ingest_ready = (
        statuses.get("verify_ingest") == "ready"
        and statuses.get("generate_installer_config") == "ready"
        and statuses.get("synthetic_capture") in {"ready", "skipped"}
        and installer_smoke_ok
    )
    worker_upload_ready = statuses.get("worker_upload_credentials") == "ready" and installer_smoke_ok
    ready = installer_smoke_ok and (local_ingest_ready or worker_upload_ready)
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "game": GAME_ID,
        "server": server,
        "output_dir": str(output_dir),
        "token": _redact_token(token),
        "ready": ready,
        "ready_paths": {
            "local_ingest": local_ingest_ready,
            "worker_upload": worker_upload_ready,
        },
        "steps": [step.to_json() for step in steps],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Ingest base URL.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "dist" / "capture" / "cyberpunk-readiness"),
        help="Directory for generated installer config, token store, and report.",
    )
    parser.add_argument("--token", default=os.environ.get("OSS_CAPTURE_INSTALL_TOKEN"))
    parser.add_argument("--mode", choices=("trickle", "lite", "regular", "INSANE"), default="lite")
    parser.add_argument("--frames", type=int, default=1, help="Synthetic fallback frames to upload.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP/process readiness timeout in seconds.")
    parser.add_argument(
        "--credentials-env",
        default=str(DEFAULT_CREDENTIALS_ENV),
        help="Path to gitignored R2/Worker credential env file.",
    )
    parser.add_argument(
        "--skip-worker-upload-check",
        action="store_true",
        help="Do not perform the authenticated Worker upload smoke check.",
    )
    parser.add_argument(
        "--worker-smoke-key",
        default="",
        help="Optional Worker smoke key. Must be allowed by the Worker, e.g. test/capture-ingest-smoke.json.",
    )
    parser.add_argument("--installer-version", default="1.0.0")
    parser.add_argument("--start-ingest", action="store_true", help="Start local loopback ingest if not already running.")
    parser.add_argument(
        "--keep-ingest-running",
        action="store_true",
        help="Leave a process started by this harness running after the report.",
    )
    parser.add_argument(
        "--mint-local-token",
        action="store_true",
        help="Generate a local token in output-dir/token-store.json when --token is absent.",
    )
    parser.add_argument("--cyberpunk-exe", default=None, help="Path to Cyberpunk2077.exe.")
    parser.add_argument("--launch-cyberpunk", action="store_true", help="Launch Cyberpunk with benchmark args if found.")
    parser.add_argument(
        "--smoke-fake-cyberpunk-install",
        action="store_true",
        help=(
            "If WiX v3/msiexec are available, build the MSI and install/uninstall "
            "it against a temporary fake Cyberpunk bin\\x64 tree."
        ),
    )
    parser.add_argument(
        "--keep-smoke-workdir",
        action="store_true",
        help="Keep the temporary installer smoke directory for inspection.",
    )
    parser.add_argument(
        "--benchmark-args",
        nargs="*",
        default=list(DEFAULT_BENCHMARK_ARGS),
        help="Arguments passed to Cyberpunk when --launch-cyberpunk is set.",
    )
    parser.add_argument("--report", default="", help="Optional path to write the JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token_store = output_dir / "token-store.json"
    credentials_env = Path(args.credentials_env).expanduser().resolve()
    runtime_env = build_runtime_env(credentials_env=credentials_env)
    token: str | None = args.token
    steps: list[StepResult] = []
    started_proc: subprocess.Popen[str] | None = None

    try:
        preflight = credential_preflight(env=runtime_env, credentials_env=credentials_env)
        steps.append(preflight)
        if preflight.status in {"worker_upload", "both"} and not args.skip_worker_upload_check:
            steps.append(
                verify_worker_upload_credentials(
                    env=runtime_env,
                    timeout=min(args.timeout, 10.0),
                    smoke_key=args.worker_smoke_key or None,
                )
            )

        if not token and args.mint_local_token:
            token, step = mint_local_token(token_store)
            steps.append(step)

        started_proc, start_step = maybe_start_ingest(
            args.server,
            token_store=token_store,
            env=runtime_env,
            start=args.start_ingest,
            keep_running=args.keep_ingest_running,
            timeout=args.timeout,
        )
        steps.append(start_step)

        steps.append(probe_ingest(args.server, timeout=min(args.timeout, 5.0)))
        steps.append(
            generate_installer_inputs(
                output_dir,
                server=args.server,
                token=token,
                mode=args.mode,
                installer_version=args.installer_version,
            )
        )
        if args.smoke_fake_cyberpunk_install:
            steps.append(
                smoke_fake_cyberpunk_installer(
                    output_dir,
                    keep_workdir=args.keep_smoke_workdir,
                )
            )
        steps.append(
            maybe_launch_cyberpunk(
                explicit_exe=args.cyberpunk_exe,
                launch=args.launch_cyberpunk,
                benchmark_args=args.benchmark_args,
            )
        )
        steps.append(
            run_synthetic_capture(
                server=args.server,
                token=token,
                frames=args.frames,
                mode=args.mode,
                timeout=args.timeout,
            )
        )
    finally:
        if started_proc is not None and not args.keep_ingest_running:
            started_proc.terminate()
            try:
                started_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                started_proc.kill()
                started_proc.wait(timeout=5.0)
            steps.append(
                StepResult(
                    "stop_ingest",
                    "ready",
                    {"pid": started_proc.pid, "returncode": started_proc.returncode},
                )
            )

    report = build_readiness_report(
        steps=steps,
        server=args.server,
        output_dir=output_dir,
        token=token,
    )
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).expanduser().write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

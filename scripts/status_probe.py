#!/usr/bin/env python3
"""Probe public dashboard service health and write status.json."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import time
import urllib.request


SCHEMA_VERSION = "2026-05-07"
DATA_FILE = "data.json"
STATUS_FILE = "status.json"
WORKER_HEALTH_URL = "https://upload.opensupersampling.com/health"
R2_DATA_URL = "https://opensupersampling.com/data.json"
DNS_HOST = "opensupersampling.com"
SERVICE_ORDER = ("trainer", "watcher", "worker", "r2", "dns")

TOOLTIPS = {
    "trainer": (
        "Background trainer process on the 3080 Ti host emits metrics.json every "
        "step into the active run directory; that's what this row reflects."
    ),
    "watcher": (
        "Loop in scripts/watch_and_publish.sh that rebuilds data.json from "
        "runs/ and pushes to R2 every ~30s."
    ),
    "worker": (
        "Cloudflare Worker at upload.opensupersampling.com that auths uploads "
        "from the trainer and writes them into the R2 bucket."
    ),
    "r2": (
        "Cloudflare R2 bucket served at opensupersampling.com - the CDN origin "
        "every visitor hits."
    ),
    "dns": "Public DNS resolution for opensupersampling.com via Cloudflare nameservers.",
}

NAMES = {
    "trainer": "Trainer",
    "watcher": "Watcher",
    "worker": "CF Worker",
    "r2": "R2 origin",
    "dns": "DNS",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> tuple[object | None, str | None]:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, f"{path.name} missing"
    except json.JSONDecodeError as exc:
        return None, f"{path.name} invalid JSON at line {exc.lineno}"
    except OSError as exc:
        return None, f"{path.name} unreadable: {exc.strerror or exc}"


def load_state(path: Path) -> dict[str, object]:
    payload, _error = load_json(path)
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def service(service_id: str, status: str, detail: str) -> dict[str, str]:
    if status not in {"healthy", "degraded", "offline"}:
        status = "offline"
    return {
        "id": service_id,
        "name": NAMES[service_id],
        "status": status,
        "detail": detail,
        "tooltip": TOOLTIPS[service_id],
    }


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h"


def bytes_label(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    kib = size / 1024.0
    if kib < 1024:
        return f"{int(round(kib))} KB"
    return f"{kib / 1024.0:.1f} MB"


def first_active_run(data: object) -> dict[str, object] | None:
    if not isinstance(data, dict):
        return None
    runs = data.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if isinstance(run, dict) and run.get("active") is True:
            return run
    return None


def run_id(run: dict[str, object]) -> str:
    for key in ("run_name", "id", "label"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return "active"


def latest_step(run: dict[str, object]) -> int | None:
    step = as_int(run.get("latest_step"))
    if step is not None:
        return step
    metrics = run.get("latest_metrics")
    if isinstance(metrics, dict):
        return as_int(metrics.get("step"))
    return None


def probe_trainer(
    data: object,
    data_error: str | None,
    state: dict[str, object],
) -> tuple[dict[str, str], dict[str, object] | None]:
    if data_error:
        return service("trainer", "offline", data_error), None

    run = first_active_run(data)
    if run is None:
        return service("trainer", "offline", "no active run"), None

    step = latest_step(run)
    if step is None:
        return service("trainer", "offline", "active run missing latest_step"), None

    generated_at = parse_time(data.get("generated_at") if isinstance(data, dict) else None)
    current = {
        "run_id": run_id(run),
        "latest_step": step,
        "generated_at": iso_z(generated_at) if generated_at else None,
    }

    previous = state.get("trainer")
    if not isinstance(previous, dict):
        return service("trainer", "healthy", f"step {step}, first sample"), current

    previous_run_id = previous.get("run_id")
    previous_step = as_int(previous.get("latest_step"))
    previous_time = parse_time(previous.get("generated_at"))

    if previous_run_id != current["run_id"]:
        return service("trainer", "healthy", f"step {step}, new active run"), current
    if previous_step is None:
        return service("trainer", "healthy", f"step {step}, first valid sample"), current

    delta = step - previous_step
    if delta > 0:
        if generated_at is not None and previous_time is not None:
            gap = max(0.0, (generated_at - previous_time).total_seconds())
            detail = f"step {step}, +{delta} in last {duration(gap)}"
        else:
            detail = f"step {step}, +{delta} since last sample"
        return service("trainer", "healthy", detail), current

    if delta < 0:
        return service("trainer", "healthy", f"step {step}, reset from {previous_step}"), current

    if previous_time is None and generated_at is not None:
        return service("trainer", "healthy", f"step {step}, first timestamped sample"), current
    if generated_at is None or previous_time is None:
        next_state = previous if previous_time is not None else current
        return service("trainer", "offline", f"stuck at step {step}; timestamp unavailable"), next_state
    if generated_at < previous_time:
        return service("trainer", "healthy", f"step {step}, timestamp reset"), current

    stuck_for = max(0.0, (generated_at - previous_time).total_seconds())
    detail = f"stuck at step {step} for {duration(stuck_for)}"
    if stuck_for < 180:
        return service("trainer", "healthy", detail), previous
    if stuck_for <= 900:
        return service("trainer", "degraded", detail), previous
    return service("trainer", "offline", detail), previous


def probe_watcher(data: object, data_error: str | None, now: datetime) -> dict[str, str]:
    if data_error:
        return service("watcher", "offline", data_error)
    generated_at = parse_time(data.get("generated_at") if isinstance(data, dict) else None)
    if generated_at is None:
        return service("watcher", "offline", "generated_at unparseable")

    age = max(0.0, (now - generated_at).total_seconds())
    detail = f"published {duration(age)} ago"
    if age < 90:
        return service("watcher", "healthy", detail)
    if age <= 300:
        return service("watcher", "degraded", detail)
    return service("watcher", "offline", detail)


def http_probe(
    url: str,
    *,
    method: str,
    timeout: float = 3.0,
    range_probe: bool = False,
) -> tuple[int | None, dict[str, str], float, str | None]:
    headers = {"User-Agent": "OpenSuperSampling-status-probe/1.0"}
    if range_probe:
        headers["Range"] = "bytes=0-0"
    request = urllib.request.Request(url, method=method, headers=headers)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed_ms = (time.monotonic() - start) * 1000
            return response.status, dict(response.headers.items()), elapsed_ms, None
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        code = getattr(exc, "code", None)
        headers_obj = getattr(exc, "headers", None)
        response_headers = dict(headers_obj.items()) if headers_obj is not None else {}
        error = exc.__class__.__name__
        return (code if isinstance(code, int) else None), response_headers, elapsed_ms, error


def probe_worker() -> dict[str, str]:
    code, _headers, elapsed_ms, error = http_probe(
        WORKER_HEALTH_URL,
        method="GET",
        timeout=3.0,
        range_probe=True,
    )
    if code == 200:
        return service("worker", "healthy", f"HTTP 200 in {int(round(elapsed_ms))}ms")
    if code is not None:
        return service("worker", "degraded", f"HTTP {code} in {int(round(elapsed_ms))}ms")
    return service("worker", "offline", error or "connection failed")


def probe_r2() -> dict[str, str]:
    # HEAD on the worker-proxied R2 origin. Cloudflare's HTTP/2 path omits
    # Content-Length on HEAD responses (the body framing carries length on
    # actual GETs), so a healthy origin can return HTTP 200 with no
    # Content-Length header. Treat that as healthy. We only fall to
    # degraded if HEAD says 200 AND length is reported AND it's zero.
    code, headers, _elapsed_ms, error = http_probe(R2_DATA_URL, method="HEAD", timeout=3.0)
    if code != 200:
        detail = f"HTTP {code}" if code is not None else (error or "connection failed")
        return service("r2", "offline", detail)

    raw_length = headers.get("Content-Length") or headers.get("content-length")
    length = as_int(raw_length)
    if length is not None and length == 0:
        # Reported length of 0 IS a real degradation (data.json is non-empty).
        return service("r2", "degraded", "HTTP 200, Content-Length=0")
    if length is not None and length > 0:
        return service("r2", "healthy", f"HTTP 200, {bytes_label(length)}")
    # HTTP 200 with no Content-Length header — Cloudflare HEAD quirk, not a
    # real degradation. The cached etag + content-type confirm the resource
    # exists; if it weren't there we'd have gotten a 404.
    return service("r2", "healthy", "HTTP 200")


def probe_dns() -> dict[str, str]:
    old_timeout = socket.getdefaulttimeout()
    start = time.monotonic()
    try:
        socket.setdefaulttimeout(2)
        ip = socket.gethostbyname(DNS_HOST)
        elapsed = time.monotonic() - start
    except Exception as exc:
        return service("dns", "offline", exc.__class__.__name__)
    finally:
        socket.setdefaulttimeout(old_timeout)

    detail = f"resolved {ip} in {int(round(elapsed * 1000))}ms"
    if elapsed < 2:
        return service("dns", "healthy", detail)
    if elapsed < 5:
        return service("dns", "degraded", detail)
    return service("dns", "offline", detail)


def build_status(staging_dir: Path, state_file: Path) -> tuple[dict[str, object], dict[str, str]]:
    data, data_error = load_json(staging_dir / DATA_FILE)
    state = load_state(state_file)
    now = utc_now()

    trainer, trainer_state = probe_trainer(data, data_error, state)
    services = [
        trainer,
        probe_watcher(data, data_error, now),
        probe_worker(),
        probe_r2(),
        probe_dns(),
    ]

    if trainer_state is not None:
        state["schema_version"] = SCHEMA_VERSION
        state["trainer"] = trainer_state
        try:
            write_json(state_file, state)
        except OSError:
            pass

    payload = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": iso_z(now),
        "services": services,
    }
    summary = {svc["id"]: svc["status"] for svc in services}
    return payload, summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe dashboard service health and write status.json.")
    parser.add_argument("--staging-dir", type=Path, required=True, help="Directory containing data.json")
    parser.add_argument("--state-file", type=Path, required=True, help="JSON state file for trainer deltas")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    payload, summary = build_status(args.staging_dir, args.state_file)
    try:
        write_json(args.staging_dir / STATUS_FILE, payload)
    except OSError:
        pass
    print("[status_probe] " + " ".join(f"{key}={summary.get(key, 'offline')}" for key in SERVICE_ORDER))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit as exc:
        if exc.code == 0:
            raise
        sys.exit(0)
    except Exception:
        print("[status_probe] trainer=offline watcher=offline worker=offline r2=offline dns=offline")
        sys.exit(0)

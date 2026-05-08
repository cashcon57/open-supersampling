#!/usr/bin/env python3
"""Operator harness for synthetic OSS capture uploads.

The simulator writes the same pending-dir layout as the capture DLL/uploader,
then drains it through :func:`oss.capture.uploader.drain_once` against a
running ingest server. Reports are emitted as JSON for shell automation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oss.capture import uploader  # noqa: E402
from oss.capture.uploader import CaptureFrame, UploadConfig, UploadResult  # noqa: E402
from tests.capture.test_fixtures import make_synthetic_capture  # noqa: E402


DEFAULT_SERVER = "https://capture.oss-supersampling.dev"
VALID_MODES = ("trickle", "lite", "regular", "INSANE")


@dataclass
class UploadEvent:
    frame_uuid: str | None
    status_code: int | None
    duration_ms: float
    terminal: bool
    retryable: bool
    response: dict[str, Any] | None = None
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "frame_uuid": self.frame_uuid,
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 3),
            "terminal": self.terminal,
            "retryable": self.retryable,
        }
        if self.response is not None:
            data["response"] = self.response
        if self.message:
            data["message"] = self.message
        return data


def ingest_url(server: str) -> str:
    """Return a concrete /ingest URL from either a base URL or endpoint URL."""

    parsed = urlparse(server)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"server must be an absolute URL: {server!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/ingest") or path == "/ingest":
        return server
    path = f"{path}/ingest" if path else "/ingest"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def _read_frame_uuid(frame: CaptureFrame) -> str | None:
    try:
        meta = json.loads(frame.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = meta.get("frame_uuid")
    return str(value) if value is not None else None


def _decode_response(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def post_frame_observed(
    frame: CaptureFrame,
    target_url: str,
    install_token: str,
    events: list[UploadEvent],
    *,
    timeout: float = 30.0,
) -> UploadResult:
    """Post one frame and record status, parsed JSON, and wall timing."""

    frame_uuid = _read_frame_uuid(frame)
    boundary = f"oss-capture-{uuid.uuid4().hex}"
    body = uploader._multipart_body(frame, boundary)  # noqa: SLF001 - harness wraps uploader wire format.
    req = request.Request(
        target_url,
        data=body,
        headers={
            "Authorization": f"Bearer {install_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "oss-capture-simulator/1.0.0",
        },
        method="POST",
    )

    started = time.perf_counter()
    response_body = b""
    retry_after_seconds: float | None = None
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            retry_after_seconds = uploader._parse_retry_after(resp.headers.get("Retry-After"))  # noqa: SLF001
            response_body = resp.read()
    except error.HTTPError as exc:
        status = int(exc.code)
        retry_after_seconds = uploader._parse_retry_after(exc.headers.get("Retry-After"))  # noqa: SLF001
        response_body = exc.read()
    except (OSError, TimeoutError) as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        result = UploadResult(None, terminal=False, retryable=True, message=str(exc))
        events.append(
            UploadEvent(
                frame_uuid=frame_uuid,
                status_code=None,
                duration_ms=duration_ms,
                terminal=False,
                retryable=True,
                message=str(exc),
            )
        )
        return result

    duration_ms = (time.perf_counter() - started) * 1000.0
    if 200 <= status < 300:
        result = UploadResult(status, terminal=True, retryable=False)
    elif status == 429:
        result = UploadResult(
            status,
            terminal=False,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )
    elif 400 <= status < 500:
        result = UploadResult(status, terminal=True, retryable=False)
    else:
        result = UploadResult(status, terminal=False, retryable=True)

    events.append(
        UploadEvent(
            frame_uuid=frame_uuid,
            status_code=status,
            duration_ms=duration_ms,
            terminal=result.terminal,
            retryable=result.retryable,
            response=_decode_response(response_body),
        )
    )
    return result


def build_simulation_report(
    *,
    game: str,
    frames_requested: int,
    mode: str,
    target_url: str,
    frames_sent: int,
    events: Sequence[UploadEvent],
    elapsed_ms: float,
) -> dict[str, Any]:
    accepts = [event for event in events if event.status_code is not None and 200 <= event.status_code < 300]
    dedup_hits = [event for event in events if event.status_code == 409]
    retryable = [event for event in events if event.retryable]
    accepted_keys = [
        event.response.get("exr_key")
        for event in accepts
        if event.response is not None and event.response.get("exr_key")
    ]
    return {
        "game": game,
        "mode": mode,
        "server": target_url,
        "frames_requested": frames_requested,
        "frames_sent": frames_sent,
        "accepts": len(accepts),
        "dedup_hits": len(dedup_hits),
        "retryable_responses": len(retryable),
        "server_timings": {
            "total_ms": round(elapsed_ms, 3),
            "requests": [event.to_json() for event in events],
        },
        "accepted_exr_keys": accepted_keys,
    }


def simulate_session(args: argparse.Namespace) -> int:
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    target_url = ingest_url(args.server)
    session_uuid = str(uuid.uuid4())
    events: list[UploadEvent] = []
    with tempfile.TemporaryDirectory(prefix="oss-capture-sim-") as tmp:
        pending = Path(tmp) / "pending"
        for index in range(args.frames):
            make_synthetic_capture(
                pending,
                game_id=args.game,
                game_version=args.game_version,
                session_uuid=session_uuid,
                frame_uuid=str(uuid.uuid4()),
                burst_uuid=session_uuid,
                burst_index=index,
                capture_mode=args.mode,
                lr_resolution=(16 + index, 9),
            )

        config = UploadConfig(
            pending_dir=pending,
            ingest_url=target_url,
            install_token=args.token,
            max_attempts=args.max_attempts,
            backoff_seconds=tuple(args.backoff),
        )
        started = time.perf_counter()
        frames_sent = uploader.drain_once(
            config,
            post=lambda frame, url, token: post_frame_observed(
                frame,
                url,
                token,
                events,
                timeout=args.timeout,
            ),
            sleep=time.sleep,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    report = build_simulation_report(
        game=args.game,
        frames_requested=args.frames,
        mode=args.mode,
        target_url=target_url,
        frames_sent=frames_sent,
        events=events,
        elapsed_ms=elapsed_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepts"] >= 1 else 1


def verify_r2(args: argparse.Namespace) -> int:
    from server.oss_capture_ingest.r2 import R2Config, R2Client

    cfg = R2Config.from_env()
    cfg.bucket = args.bucket
    client = R2Client(cfg)
    objects = [
        {"key": key, "size": size}
        for key, size in client.iter_objects(prefix=args.prefix)
        if not key.startswith("_dedup/")
    ]
    exr_count = sum(1 for obj in objects if str(obj["key"]).endswith(".exr"))
    json_count = sum(1 for obj in objects if str(obj["key"]).endswith(".json"))
    report = {
        "bucket": args.bucket,
        "prefix": args.prefix,
        "objects": objects,
        "object_count": len(objects),
        "exr_count": exr_count,
        "json_count": json_count,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if exr_count > 0 else 1


def mint_test_token(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "server.oss_capture_ingest.main",
        "mint-token",
        "--label",
        args.label,
    ]
    if args.token:
        cmd.extend(["--token", args.token])
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate-session", help="Upload synthetic captures through a running ingest server.")
    p_sim.add_argument("--game", required=True, help="Game id to write into metadata, e.g. cyberpunk-2077.")
    p_sim.add_argument("--frames", type=int, required=True, help="Number of synthetic frames to enqueue.")
    p_sim.add_argument("--mode", choices=VALID_MODES, required=True, help="Capture mode to write into metadata.")
    p_sim.add_argument("--server", default=DEFAULT_SERVER, help="Ingest base URL or full /ingest URL.")
    p_sim.add_argument("--token", required=True, help="Install bearer token minted by the ingest server.")
    p_sim.add_argument("--game-version", default="sim-test", help="Synthetic game_version metadata value.")
    p_sim.add_argument("--max-attempts", type=int, default=1, help="Uploader retry attempts per frame.")
    p_sim.add_argument("--backoff", type=float, nargs="*", default=[0.0], help="Retry backoff sequence in seconds.")
    p_sim.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout per upload attempt.")
    p_sim.set_defaults(func=simulate_session)

    p_r2 = sub.add_parser("verify-r2", help="List accepted capture objects in R2.")
    p_r2.add_argument("--bucket", required=True, help="R2 bucket name, e.g. ors-captures.")
    p_r2.add_argument("--prefix", default="", help="R2 key prefix, e.g. cyberpunk-2077/.")
    p_r2.set_defaults(func=verify_r2)

    p_token = sub.add_parser("mint-test-token", help="Mint a test token via the ingest-server CLI.")
    p_token.add_argument("--label", default="sim-test", help="Human-readable token label.")
    p_token.add_argument("--token", default=None, help="Use this exact token instead of generating one.")
    p_token.set_defaults(func=mint_test_token)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

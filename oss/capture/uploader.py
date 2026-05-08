"""Drain locally captured OSS training frames to the ingest service.

Status handling is intentionally conservative for terminal outcomes: 2xx
responses, revoked auth (401), durable dedup hits (409), and oversize payloads
(413) delete local files because there is no useful local retry. Other 4xx
responses are terminal as well. Rate limits (429) are retryable and stay pending
after this pass, while 5xx and network errors retry with backoff and are dropped
only after exhausting configured attempts.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import logging
import mimetypes
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence
from urllib import error, request


LOGGER = logging.getLogger("oss.capture.uploader")
DEFAULT_INGEST_URL = "https://capture.oss-supersampling.dev/ingest"
DEFAULT_MAX_PENDING_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_BACKOFF_SECONDS = (1.0, 5.0, 30.0, 120.0, 600.0)


@dataclass(frozen=True)
class UploadConfig:
    pending_dir: Path
    ingest_url: str
    install_token: str
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    max_attempts: int = 5
    scan_interval_seconds: float = 60.0
    backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS


@dataclass(frozen=True)
class CaptureFrame:
    frame_path: Path
    meta_path: Path


@dataclass(frozen=True)
class UploadResult:
    status_code: int | None
    terminal: bool
    retryable: bool
    message: str = ""
    retry_after_seconds: float | None = None


def default_capture_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "oss-capture"
    return Path.home() / "AppData" / "Local" / "oss-capture"


def load_config(path: Path | None = None) -> UploadConfig:
    root = default_capture_root()
    config_path = path or root / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    pending_dir = Path(raw.get("pending_dir", root / "pending"))
    return UploadConfig(
        pending_dir=pending_dir,
        ingest_url=str(raw.get("ingest_url", DEFAULT_INGEST_URL)),
        install_token=str(raw["install_token"]),
        max_pending_bytes=int(raw.get("max_pending_bytes", DEFAULT_MAX_PENDING_BYTES)),
        max_attempts=int(raw.get("max_attempts", 5)),
        scan_interval_seconds=float(raw.get("scan_interval_seconds", 60.0)),
    )


def iter_frames(pending_dir: Path) -> Iterator[CaptureFrame]:
    if not pending_dir.exists():
        return
    for frame_path in sorted(pending_dir.glob("*/*/*.exr")):
        meta_path = frame_path.with_suffix(".json")
        if meta_path.exists():
            yield CaptureFrame(frame_path=frame_path, meta_path=meta_path)
        else:
            LOGGER.warning("dropping orphan frame without metadata: %s", frame_path)
            delete_frame_pair(CaptureFrame(frame_path=frame_path, meta_path=meta_path))


def delete_frame_pair(frame: CaptureFrame) -> None:
    for path in (frame.frame_path, frame.meta_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def pending_bytes(pending_dir: Path) -> int:
    if not pending_dir.exists():
        return 0
    total = 0
    for path in pending_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _captured_at_unix(meta_path: Path) -> float | None:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(raw["captured_at_unix"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _oldest_frame_key(frame: CaptureFrame) -> tuple[int, float, int, int, str, str]:
    frame_stat = frame.frame_path.stat()
    meta_stat = frame.meta_path.stat()
    fs_mtime_ns = min(frame_stat.st_mtime_ns, meta_stat.st_mtime_ns)
    fs_ctime_ns = min(frame_stat.st_ctime_ns, meta_stat.st_ctime_ns)
    captured_at = _captured_at_unix(frame.meta_path)
    if captured_at is not None:
        return (0, captured_at, fs_mtime_ns, fs_ctime_ns, frame.frame_path.name, str(frame.frame_path))
    return (1, float(fs_mtime_ns), fs_mtime_ns, fs_ctime_ns, frame.frame_path.name, str(frame.frame_path))


def enforce_pending_cap(pending_dir: Path, max_bytes: int) -> list[Path]:
    """Delete oldest frame pairs until pending_dir is at or below max_bytes."""

    deleted: list[Path] = []
    total = pending_bytes(pending_dir)
    if total <= max_bytes:
        return deleted

    candidates = sorted(
        iter_frames(pending_dir),
        key=_oldest_frame_key,
    )
    for frame in candidates:
        size = frame.frame_path.stat().st_size
        size += frame.meta_path.stat().st_size if frame.meta_path.exists() else 0
        delete_frame_pair(frame)
        deleted.append(frame.frame_path)
        total -= size
        LOGGER.warning("pending cap exceeded; dropped oldest capture: %s", frame.frame_path)
        if total <= max_bytes:
            break
    return deleted


def _multipart_body(frame: CaptureFrame, boundary: str) -> bytes:
    meta_bytes = frame.meta_path.read_bytes()
    frame_bytes = frame.frame_path.read_bytes()
    frame_name = frame.frame_path.name
    frame_type = mimetypes.guess_type(frame_name)[0] or "application/octet-stream"

    parts: list[bytes] = []
    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="meta"\r\n'
            "Content-Type: application/json\r\n\r\n"
        ).encode("utf-8")
        + meta_bytes
        + b"\r\n"
    )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="frame"; filename="{frame_name}"\r\n'
            f"Content-Type: {frame_type}\r\n\r\n"
        ).encode("utf-8")
        + frame_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)


def _parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)
    return max(0.0, (retry_at - now).total_seconds())


def post_frame(frame: CaptureFrame, ingest_url: str, install_token: str, timeout: float = 30.0) -> UploadResult:
    boundary = f"oss-capture-{uuid.uuid4().hex}"
    body = _multipart_body(frame, boundary)
    req = request.Request(
        ingest_url,
        data=body,
        headers={
            "Authorization": f"Bearer {install_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "oss-capture-uploader/1.0.0",
        },
        method="POST",
    )

    retry_after_seconds: float | None = None
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            retry_after_seconds = _parse_retry_after(resp.headers.get("Retry-After"))
    except error.HTTPError as exc:
        status = int(exc.code)
        retry_after_seconds = _parse_retry_after(exc.headers.get("Retry-After"))
    except (OSError, TimeoutError) as exc:
        return UploadResult(None, terminal=False, retryable=True, message=str(exc))

    if 200 <= status < 300:
        return UploadResult(status, terminal=True, retryable=False)
    if status == 429:
        return UploadResult(
            status,
            terminal=False,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )
    if 400 <= status < 500:
        return UploadResult(status, terminal=True, retryable=False)
    return UploadResult(status, terminal=False, retryable=True)


def upload_with_retries(
    frame: CaptureFrame,
    config: UploadConfig,
    *,
    post: Callable[[CaptureFrame, str, str], UploadResult] = post_frame,
    sleep: Callable[[float], None] = time.sleep,
) -> UploadResult:
    attempts = max(1, config.max_attempts)
    last = UploadResult(None, terminal=False, retryable=True)
    for attempt in range(attempts):
        last = post(frame, config.ingest_url, config.install_token)
        if last.terminal:
            delete_frame_pair(frame)
            return last
        if attempt < attempts - 1:
            base_delay = config.backoff_seconds[min(attempt, len(config.backoff_seconds) - 1)]
            if last.status_code == 429 and last.retry_after_seconds is not None:
                delay = last.retry_after_seconds
            elif last.status_code == 429:
                delay = base_delay * 4.0
            else:
                delay = base_delay
            sleep(delay)

    if last.status_code == 429:
        LOGGER.warning("leaving rate-limited capture pending for next upload pass: %s", frame.frame_path)
        return last

    LOGGER.warning(
        "dropping capture after exhausted upload retries: %s status=%s message=%s",
        frame.frame_path,
        last.status_code,
        last.message,
    )
    delete_frame_pair(frame)
    return UploadResult(last.status_code, terminal=True, retryable=False, message=last.message)


def drain_once(
    config: UploadConfig,
    *,
    post: Callable[[CaptureFrame, str, str], UploadResult] = post_frame,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    config.pending_dir.mkdir(parents=True, exist_ok=True)
    enforce_pending_cap(config.pending_dir, config.max_pending_bytes)
    uploaded_or_dropped = 0
    for frame in list(iter_frames(config.pending_dir)):
        upload_with_retries(frame, config, post=post, sleep=sleep)
        uploaded_or_dropped += 1
    prune_empty_dirs(config.pending_dir)
    return uploaded_or_dropped


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def run_forever(config: UploadConfig) -> None:
    while True:
        drain_once(config)
        time.sleep(config.scan_interval_seconds + random.uniform(0.0, 1.0))


def _parse_backoff(values: str | None) -> tuple[float, ...]:
    if not values:
        return DEFAULT_BACKOFF_SECONDS
    return tuple(float(value) for value in values.split(",") if value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to oss-capture config.json")
    parser.add_argument("--pending-dir", type=Path, help="Override pending capture directory")
    parser.add_argument("--ingest-url", help="Override ingest endpoint")
    parser.add_argument("--install-token", help="Override bearer token")
    parser.add_argument("--once", action="store_true", help="Drain pending captures once and exit")
    parser.add_argument("--scan-interval", type=float, default=None, help="Seconds between scans")
    parser.add_argument("--max-attempts", type=int, default=None, help="Retries before dropping a frame")
    parser.add_argument("--max-pending-bytes", type=int, default=None, help="Pending directory hard cap")
    parser.add_argument("--backoff", help="Comma-separated retry delays in seconds")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def config_from_args(args: argparse.Namespace) -> UploadConfig:
    loaded: UploadConfig | None = None
    if args.config:
        loaded = load_config(args.config)

    root = default_capture_root()
    pending_dir = args.pending_dir or (loaded.pending_dir if loaded else root / "pending")
    ingest_url = args.ingest_url or (loaded.ingest_url if loaded else DEFAULT_INGEST_URL)
    install_token = args.install_token or (loaded.install_token if loaded else None)
    if not install_token:
        raise SystemExit("--install-token or --config with install_token is required")

    return UploadConfig(
        pending_dir=pending_dir,
        ingest_url=ingest_url,
        install_token=install_token,
        max_pending_bytes=args.max_pending_bytes
        if args.max_pending_bytes is not None
        else (loaded.max_pending_bytes if loaded else DEFAULT_MAX_PENDING_BYTES),
        max_attempts=args.max_attempts if args.max_attempts is not None else (loaded.max_attempts if loaded else 5),
        scan_interval_seconds=args.scan_interval
        if args.scan_interval is not None
        else (loaded.scan_interval_seconds if loaded else 60.0),
        backoff_seconds=_parse_backoff(args.backoff),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = config_from_args(args)
    if args.once:
        drain_once(config)
    else:
        run_forever(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

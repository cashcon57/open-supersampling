"""Drain locally captured OSS training frames to the ingest service.

Status handling is intentionally conservative for terminal outcomes: 2xx
responses, revoked auth (401), durable dedup hits (409), and oversize payloads
(413) delete local files because there is no useful local retry. Other 4xx
responses are terminal as well. Rate limits (429) are retryable and stay pending
after this pass. Server 5xx and network errors retry with backoff, then stay
pending for a later scheduled pass; the pending-dir cap is the only eviction
mechanism for retryable failures.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import logging
import math
import mimetypes
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib import error, request


LOGGER = logging.getLogger("oss.capture.uploader")
DEFAULT_INGEST_URL = "https://capture.oss-supersampling.dev/ingest"
DEFAULT_MAX_PENDING_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_BACKOFF_SECONDS = (1.0, 5.0, 30.0, 120.0, 600.0)
DEFAULT_INCOMPLETE_GRACE_SECONDS = 120.0
REQUIRED_DURABLE_METADATA_FIELDS = (
    "capture_kind",
    "provider",
    "captured_at_monotonic_seconds",
    "sequence_index",
    "render_resolution",
    "output_resolution",
    "dxgi_formats",
    "jitter_offset_px",
    "exposure_scale",
    "channel_presence",
)


@dataclass(frozen=True)
class UploadConfig:
    pending_dir: Path
    ingest_url: str
    install_token: str
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    max_attempts: int = 5
    scan_interval_seconds: float = 60.0
    backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS
    incomplete_grace_seconds: float = DEFAULT_INCOMPLETE_GRACE_SECONDS
    capture_storage_mode: str = "local"


@dataclass(frozen=True)
class CaptureFrame:
    frame_path: Path
    meta_path: Path


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    message: str


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
    capture_storage_mode = str(raw.get("capture_storage_mode", "local"))
    if capture_storage_mode not in {"local", "upload"}:
        raise ValueError(f"unknown capture_storage_mode {capture_storage_mode!r}")
    pending_dir = Path(raw.get("pending_dir", root / "pending"))
    endpoints = raw.get("endpoints") if isinstance(raw.get("endpoints"), dict) else {}
    ingest_url = raw.get("ingest_url") or endpoints.get("ingest")
    if not ingest_url and raw.get("capture_api_base"):
        ingest_url = str(raw["capture_api_base"]).rstrip("/") + "/ingest"
    return UploadConfig(
        pending_dir=pending_dir,
        ingest_url=str(ingest_url or DEFAULT_INGEST_URL),
        install_token=str(raw["install_token"]),
        capture_storage_mode=capture_storage_mode,
        max_pending_bytes=int(
            raw.get("max_pending_bytes", raw.get("pending_dir_cap_bytes", DEFAULT_MAX_PENDING_BYTES))
        ),
        max_attempts=int(raw.get("max_attempts", raw.get("uploader_retry_attempts", 5))),
        scan_interval_seconds=float(raw.get("scan_interval_seconds", 60.0)),
        incomplete_grace_seconds=float(
            raw.get("incomplete_grace_seconds", DEFAULT_INCOMPLETE_GRACE_SECONDS)
        ),
    )


def _newest_mtime_age_seconds(paths: Sequence[Path], *, now: float | None = None) -> float | None:
    mtimes: list[float] = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    if not mtimes:
        return None
    if now is None:
        now = time.time()
    return max(0.0, now - max(mtimes))


def _is_incomplete_frame_young(
    frame: CaptureFrame,
    *,
    grace_seconds: float,
    now: float | None = None,
) -> bool:
    age = _newest_mtime_age_seconds((frame.frame_path, frame.meta_path), now=now)
    return age is not None and age < max(0.0, grace_seconds)


def iter_frames(
    pending_dir: Path,
    *,
    incomplete_grace_seconds: float = DEFAULT_INCOMPLETE_GRACE_SECONDS,
    now: float | None = None,
) -> Iterator[CaptureFrame]:
    if not pending_dir.exists():
        return
    for frame_path in sorted(pending_dir.glob("*/*/*.exr")):
        meta_path = frame_path.with_suffix(".json")
        if meta_path.exists():
            yield CaptureFrame(frame_path=frame_path, meta_path=meta_path)
        elif _is_incomplete_frame_young(
            CaptureFrame(frame_path=frame_path, meta_path=meta_path),
            grace_seconds=incomplete_grace_seconds,
            now=now,
        ):
            LOGGER.debug("leaving young orphan frame pending for metadata writer: %s", frame_path)
        else:
            LOGGER.warning("dropping orphan frame without metadata: %s", frame_path)
            delete_frame_pair(CaptureFrame(frame_path=frame_path, meta_path=meta_path))


def delete_frame_pair(frame: CaptureFrame) -> None:
    for path in (frame.frame_path, frame.meta_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _read_metadata(meta_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"metadata unreadable: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"metadata is not valid JSON: {exc}"
    if not isinstance(raw, dict):
        return None, "metadata must be a JSON object"
    return raw, None


def _resolution_pair(meta: dict[str, Any], key: str) -> tuple[int, int] | None:
    value = meta.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        return None
    return int(value[0]), int(value[1])


def _required_exr_groups(meta: dict[str, Any]) -> dict[str, int]:
    presence = meta.get("channel_presence")
    if not isinstance(presence, dict):
        presence = {}
    groups: dict[str, int] = {}
    if presence.get("lr", True):
        groups["LR"] = 3
    if presence.get("depth", True):
        groups["Depth"] = 1
    if presence.get("motion", True):
        groups["Motion"] = 2
    if presence.get("normals", False):
        groups["Normals"] = 3
    if meta.get("hr_source") != "none" and presence.get("hr", True):
        groups["HR"] = 3
    if presence.get("albedo", False):
        groups["Albedo"] = 3
    if presence.get("roughness", False):
        groups["Roughness"] = 1
    if presence.get("metallic", False):
        groups["Metallic"] = 1
    if presence.get("emissive", False):
        groups["Emissive"] = 3
    return groups


def _exr_validation_error(frame: CaptureFrame, meta: dict[str, Any]) -> str | None:
    hr_resolution = _resolution_pair(meta, "hr_resolution")
    lr_resolution = _resolution_pair(meta, "lr_resolution")
    if hr_resolution is None:
        return "metadata hr_resolution must be [width, height] of positive ints"
    if lr_resolution is None:
        return "metadata lr_resolution must be [width, height] of positive ints"
    if lr_resolution[0] > hr_resolution[0] or lr_resolution[1] > hr_resolution[1]:
        return "metadata lr_resolution must not exceed hr_resolution"

    try:
        import numpy as np
        import pyexr
    except Exception as exc:  # pragma: no cover - exercised only in stripped installs.
        return f"EXR validation dependency unavailable: {exc}"

    try:
        with pyexr.open(frame.frame_path) as exr:
            channels = set(exr.channels)
            for group, expected_components in _required_exr_groups(meta).items():
                try:
                    data = exr.get(group)
                except Exception as exc:
                    return f"EXR missing required channel group {group!r}: {exc}"
                if data is None:
                    return f"EXR missing required channel group {group!r}"
                if data.ndim == 2:
                    data = data[..., None]
                if data.ndim != 3 or data.shape[2] != expected_components:
                    return (
                        f"EXR channel group {group!r} has shape {tuple(data.shape)}, "
                        f"expected HxWx{expected_components}"
                    )
                if tuple(data.shape[:2]) != (hr_resolution[1], hr_resolution[0]):
                    return (
                        f"EXR channel group {group!r} dimensions "
                        f"{data.shape[1]}x{data.shape[0]} do not match "
                        f"metadata hr_resolution {hr_resolution[0]}x{hr_resolution[1]}"
                    )
                if not np.isfinite(data).all():
                    return f"EXR channel group {group!r} contains NaN or Inf"
                # Static captures may legitimately have zero motion. The other
                # core training signals should not be blank.
                if group != "Motion" and not np.any(data != 0):
                    return f"EXR channel group {group!r} is all zero"

            expected_names = {
                "LR": ("R", "G", "B"),
                "HR": ("R", "G", "B"),
                "Depth": ("Z",),
                "Motion": ("X", "Y"),
                "Normals": ("X", "Y", "Z"),
                "Albedo": ("R", "G", "B"),
                "Roughness": ("R",),
                "Metallic": ("R",),
                "Emissive": ("R", "G", "B"),
            }
            for group in _required_exr_groups(meta):
                missing = [f"{group}.{name}" for name in expected_names[group] if f"{group}.{name}" not in channels]
                if missing:
                    return f"EXR missing required channel(s): {', '.join(missing)}"
    except OSError as exc:
        return f"EXR unreadable: {exc}"
    except Exception as exc:
        return f"EXR validation failed: {exc}"
    return None


def validate_pending_frame(
    frame: CaptureFrame,
    pending_dir: Path,
    *,
    allow_local: bool = False,
    validate_exr: bool = True,
) -> str | None:
    """Return a terminal local-validation error, or None if uploadable."""

    try:
        rel = frame.frame_path.relative_to(pending_dir)
    except ValueError:
        return f"frame is outside pending dir: {frame.frame_path}"
    parts = rel.parts
    if len(parts) != 3:
        return "pending frame path must be <game_id>/<session_uuid>/<frame_uuid>.exr"

    path_game_id, path_session_uuid, frame_name = parts
    if frame_name != frame.frame_path.name or frame.frame_path.suffix.lower() != ".exr":
        return "pending frame path must point to an .exr file"

    meta, err = _read_metadata(frame.meta_path)
    if err is not None:
        return err
    assert meta is not None

    checks = (
        ("game_id", path_game_id),
        ("session_uuid", path_session_uuid),
        ("frame_uuid", frame.frame_path.stem),
    )
    for key, expected in checks:
        value = meta.get(key)
        if not isinstance(value, str) or not value:
            return f"metadata field {key!r} must be a non-empty string"
        if value != expected:
            return (
                f"metadata field {key!r}={value!r} does not match "
                f"pending path value {expected!r}"
            )
    missing_durable = [key for key in REQUIRED_DURABLE_METADATA_FIELDS if key not in meta]
    if missing_durable:
        return "metadata missing durable field(s): " + ", ".join(missing_durable)
    storage_mode = meta.get("capture_storage_mode")
    if storage_mode not in {"local", "upload"}:
        return "metadata capture_storage_mode must be 'local' or 'upload'"
    if storage_mode == "local" and not allow_local:
        return "metadata capture_storage_mode='local' is not uploadable"
    channel_presence = meta.get("channel_presence")
    if not isinstance(channel_presence, dict):
        return "metadata channel_presence must be an object"
    hr_present = channel_presence.get("hr")
    if hr_present is not None and not isinstance(hr_present, bool):
        return "metadata channel_presence.hr must be a bool"
    if meta.get("hr_source") == "none" and hr_present is True:
        return "metadata HR channel conflict: hr_source='none' requires channel_presence.hr=false"
    if meta.get("hr_source") != "none" and hr_present is False:
        return "metadata HR channel conflict: hr_source requires channel_presence.hr=true unless hr_source='none'"
    if validate_exr:
        return _exr_validation_error(frame, meta)
    return None


def _iter_capture_candidates(pending_dir: Path) -> Iterator[CaptureFrame]:
    if not pending_dir.exists():
        return
    yielded: set[Path] = set()
    for frame_path in sorted(pending_dir.glob("*/*/*.exr")):
        yielded.add(frame_path)
        yield CaptureFrame(frame_path=frame_path, meta_path=frame_path.with_suffix(".json"))
    for meta_path in sorted(pending_dir.glob("*/*/*.json")):
        frame_path = meta_path.with_suffix(".exr")
        if frame_path not in yielded:
            yield CaptureFrame(frame_path=frame_path, meta_path=meta_path)


def _burst_validation_issues(frames: Sequence[CaptureFrame]) -> list[ValidationIssue]:
    groups: dict[tuple[str, str, str], list[tuple[int, float, CaptureFrame]]] = {}
    issues: list[ValidationIssue] = []
    for frame in frames:
        meta, err = _read_metadata(frame.meta_path)
        if err is not None or meta is None:
            continue
        burst_uuid = meta.get("burst_uuid")
        burst_index = meta.get("burst_index")
        if burst_uuid is None and burst_index is None:
            continue
        if not isinstance(burst_uuid, str) or not isinstance(burst_index, int):
            issues.append(ValidationIssue(frame.meta_path, "burst_uuid and burst_index must be valid together"))
            continue
        captured_at = meta.get("captured_at_unix")
        captured = float(captured_at) if isinstance(captured_at, (int, float)) and math.isfinite(float(captured_at)) else 0.0
        key = (str(meta.get("game_id", "")), str(meta.get("session_uuid", "")), burst_uuid)
        groups.setdefault(key, []).append((burst_index, captured, frame))

    for (_game_id, _session_uuid, burst_uuid), members in groups.items():
        if len(members) < 2:
            continue
        by_index = sorted(members, key=lambda item: item[0])
        indices = [item[0] for item in by_index]
        if len(indices) != len(set(indices)):
            for _, _, frame in members:
                issues.append(ValidationIssue(frame.meta_path, f"burst {burst_uuid} has duplicate burst_index values"))
            continue
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            missing = sorted(set(expected) - set(indices))
            for _, _, frame in members:
                issues.append(
                    ValidationIssue(
                        frame.meta_path,
                        f"burst {burst_uuid} indices are not contiguous; missing {missing}",
                    )
                )
            continue
        ordered_times = [captured for _, captured, _ in by_index]
        if any(later < earlier for earlier, later in zip(ordered_times, ordered_times[1:])):
            for _, _, frame in members:
                issues.append(
                    ValidationIssue(
                        frame.meta_path,
                        f"burst {burst_uuid} captured_at_unix decreases with burst_index",
                    )
                )
    return issues


def validate_capture_tree(
    pending_dir: Path,
    *,
    allow_local: bool = True,
    validate_exr: bool = True,
    strict_bursts: bool = True,
) -> list[ValidationIssue]:
    """Validate captured training samples without mutating the pending tree."""

    frames = list(_iter_capture_candidates(pending_dir))
    issues: list[ValidationIssue] = []
    for frame in frames:
        if not frame.frame_path.exists():
            issues.append(ValidationIssue(frame.frame_path, "missing EXR for JSON sidecar"))
            continue
        if not frame.meta_path.exists():
            issues.append(ValidationIssue(frame.frame_path, "missing JSON sidecar for EXR"))
            continue
        error = validate_pending_frame(
            frame,
            pending_dir,
            allow_local=allow_local,
            validate_exr=validate_exr,
        )
        if error is not None:
            issues.append(ValidationIssue(frame.frame_path, error))
    if strict_bursts:
        valid_frames = [
            frame
            for frame in frames
            if frame.frame_path.exists()
            and frame.meta_path.exists()
            and validate_pending_frame(
                frame,
                pending_dir,
                allow_local=allow_local,
                validate_exr=False,
            )
            is None
        ]
        issues.extend(_burst_validation_issues(valid_frames))
    return issues


def _is_local_only_frame(frame: CaptureFrame) -> bool:
    meta, err = _read_metadata(frame.meta_path)
    return err is None and meta is not None and meta.get("capture_storage_mode") == "local"


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


def _configured_backoff(config: UploadConfig, attempt: int) -> float:
    if not config.backoff_seconds:
        return 0.0
    return config.backoff_seconds[min(attempt, len(config.backoff_seconds) - 1)]


def _http_error_message(exc: error.HTTPError) -> str:
    reason = str(exc.reason or "").strip()
    try:
        body = exc.read(4096)
    except OSError:
        body = b""
    if not body:
        return reason
    decoded = body.decode("utf-8", errors="replace").strip()
    if not decoded:
        return reason
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        detail = decoded
    else:
        detail_obj = parsed.get("detail") if isinstance(parsed, dict) else parsed
        if isinstance(detail_obj, str):
            detail = detail_obj
        else:
            detail = json.dumps(detail_obj, sort_keys=True)
    return f"{reason}: {detail}" if reason else detail


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
        message = _http_error_message(exc)
    except (OSError, TimeoutError) as exc:
        return UploadResult(None, terminal=False, retryable=True, message=str(exc))
    else:
        message = ""

    if 200 <= status < 300:
        return UploadResult(status, terminal=True, retryable=False, message=message)
    if status == 429:
        return UploadResult(
            status,
            terminal=False,
            retryable=True,
            message=message,
            retry_after_seconds=retry_after_seconds,
        )
    if 400 <= status < 500:
        return UploadResult(status, terminal=True, retryable=False, message=message)
    return UploadResult(status, terminal=False, retryable=True, message=message)


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
            base_delay = _configured_backoff(config, attempt)
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
        "leaving retryable capture pending after exhausted upload retries: %s status=%s message=%s",
        frame.frame_path,
        last.status_code,
        last.message,
    )
    return last


def drain_once(
    config: UploadConfig,
    *,
    post: Callable[[CaptureFrame, str, str], UploadResult] = post_frame,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    config.pending_dir.mkdir(parents=True, exist_ok=True)
    if config.capture_storage_mode == "local":
        LOGGER.info("capture_storage_mode=local; uploader leaving captures on disk")
        return 0
    enforce_pending_cap(config.pending_dir, config.max_pending_bytes)
    uploaded_or_dropped = 0
    for frame in list(
        iter_frames(
            config.pending_dir,
            incomplete_grace_seconds=config.incomplete_grace_seconds,
        )
    ):
        if _is_local_only_frame(frame):
            LOGGER.info("leaving local-only capture on disk: %s", frame.frame_path)
            continue
        validation_error = validate_pending_frame(frame, config.pending_dir)
        if validation_error is not None:
            if _is_incomplete_frame_young(
                frame,
                grace_seconds=config.incomplete_grace_seconds,
            ):
                LOGGER.debug(
                    "leaving young incomplete pending capture for next pass: %s reason=%s",
                    frame.frame_path,
                    validation_error,
                )
                continue
            LOGGER.warning(
                "dropping invalid pending capture: %s reason=%s",
                frame.frame_path,
                validation_error,
            )
            delete_frame_pair(frame)
            uploaded_or_dropped += 1
            continue
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
    parser.add_argument(
        "--storage-mode",
        choices=("local", "upload"),
        default=None,
        help="Override capture_storage_mode. local leaves captures on disk; upload sends to ingest/R2.",
    )
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
        capture_storage_mode=args.storage_mode
        or (loaded.capture_storage_mode if loaded else "local"),
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

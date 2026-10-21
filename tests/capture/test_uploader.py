from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from email.message import Message
from pathlib import Path
from urllib import error

import numpy as np
import pytest

import oss.capture.uploader as uploader
from oss.capture.uploader import (
    CaptureFrame,
    UploadConfig,
    UploadResult,
    drain_once,
    enforce_pending_cap,
    iter_frames,
    load_config,
    post_frame,
    upload_with_retries,
    validate_capture_tree,
    validate_pending_frame,
)
from tests.capture.test_fixtures import _require_pyexr, make_synthetic_capture


def _config(pending: Path, *, incomplete_grace_seconds: float = 0.0) -> UploadConfig:
    return UploadConfig(
        pending_dir=pending,
        ingest_url="http://127.0.0.1:1/ingest",
        install_token="token",
        max_attempts=3,
        backoff_seconds=(0.0, 0.0, 0.0),
        incomplete_grace_seconds=incomplete_grace_seconds,
        capture_storage_mode="upload",
    )


def test_load_config_uses_endpoints_and_installer_knobs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "pending_dir": str(tmp_path / "captures"),
                "endpoints": {"ingest": "http://127.0.0.1:8080/ingest"},
                "install_token": "tok",
                "capture_storage_mode": "upload",
                "pending_dir_cap_bytes": 12345,
                "uploader_retry_attempts": 3,
                "incomplete_grace_seconds": 42,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.pending_dir == tmp_path / "captures"
    assert cfg.ingest_url == "http://127.0.0.1:8080/ingest"
    assert cfg.capture_storage_mode == "upload"
    assert cfg.max_pending_bytes == 12345
    assert cfg.max_attempts == 3
    assert cfg.incomplete_grace_seconds == 42


def test_load_config_defaults_to_local_storage_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "install_token": "tok",
                "capture_api_base": "http://127.0.0.1:8080",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.capture_storage_mode == "local"


def test_load_config_builds_ingest_url_from_capture_api_base(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "capture_api_base": "http://127.0.0.1:8080",
                "install_token": "tok",
            }
        ),
        encoding="utf-8",
    )

    assert load_config(config_path).ingest_url == "http://127.0.0.1:8080/ingest"


def test_load_config_local_storage_mode_disables_upload_drain(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    config_path.write_text(
        json.dumps(
            {
                "pending_dir": str(pending),
                "install_token": "tok",
                "capture_storage_mode": "local",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.capture_storage_mode == "local"
    assert drain_once(cfg, post=lambda *_: pytest.fail("local mode must not post")) == 0
    assert capture.frame_path.exists()
    assert capture.meta_path.exists()


def test_upload_mode_does_not_upload_local_only_frames(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending, capture_storage_mode="local")
    config = _config(pending)

    drained = drain_once(config, post=lambda *_: pytest.fail("local-only frame must not post"))

    assert drained == 0
    assert capture.frame_path.exists()
    assert capture.meta_path.exists()


def test_upload_with_retries_deletes_on_200_and_4xx(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    accepted = make_synthetic_capture(pending, frame_uuid="accepted")
    rejected = make_synthetic_capture(pending, frame_uuid="rejected")
    config = _config(pending)

    accepted_result = upload_with_retries(
        CaptureFrame(accepted.frame_path, accepted.meta_path),
        config,
        post=lambda *_: UploadResult(200, terminal=True, retryable=False),
        sleep=lambda _: None,
    )
    rejected_result = upload_with_retries(
        CaptureFrame(rejected.frame_path, rejected.meta_path),
        config,
        post=lambda *_: UploadResult(409, terminal=True, retryable=False),
        sleep=lambda _: None,
    )

    assert accepted_result.status_code == 200
    assert rejected_result.status_code == 409
    assert not accepted.frame_path.exists()
    assert not accepted.meta_path.exists()
    assert not rejected.frame_path.exists()
    assert not rejected.meta_path.exists()


def test_429_does_not_delete_on_first_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)

    def rate_limited(req, timeout):  # noqa: ANN001 - matches urllib hook.
        headers = Message()
        raise error.HTTPError(req.full_url, 429, "Too Many Requests", headers, None)

    monkeypatch.setattr(uploader.request, "urlopen", rate_limited)

    result = post_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        "https://example.test/ingest",
        "token",
    )

    assert result.status_code == 429
    assert result.retryable is True
    assert result.terminal is False
    assert capture.frame_path.exists()
    assert capture.meta_path.exists()


def test_429_with_retry_after_header_uses_server_hint(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    sleeps: list[float] = []
    responses = [
        UploadResult(429, terminal=False, retryable=True, retry_after_seconds=30.0),
        UploadResult(200, terminal=True, retryable=False),
    ]

    def scripted(*_: object) -> UploadResult:
        return responses.pop(0)

    result = upload_with_retries(
        CaptureFrame(capture.frame_path, capture.meta_path),
        _config(pending),
        post=scripted,
        sleep=sleeps.append,
    )

    assert result.status_code == 200
    assert sleeps == [30.0]
    assert not capture.frame_path.exists()
    assert not capture.meta_path.exists()


def test_429_after_max_attempts_stays_pending_for_next_pass(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    attempts: list[int] = []
    config = UploadConfig(
        pending_dir=pending,
        ingest_url="https://example.test/ingest",
        install_token="token",
        max_attempts=2,
        backoff_seconds=(0.0, 0.0),
        capture_storage_mode="upload",
    )

    def always_limited(*_: object) -> UploadResult:
        attempts.append(429)
        return UploadResult(429, terminal=False, retryable=True, retry_after_seconds=0.0)

    result = upload_with_retries(
        CaptureFrame(capture.frame_path, capture.meta_path),
        config,
        post=always_limited,
        sleep=lambda _: None,
    )

    assert result.status_code == 429
    assert result.terminal is False
    assert result.retryable is True
    assert attempts == [429, 429]
    assert capture.frame_path.exists()
    assert capture.meta_path.exists()


def test_upload_with_retries_leaves_5xx_pending_after_exhausted_attempts(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    attempts: list[int | None] = []

    def always_retry(*_: object) -> UploadResult:
        attempts.append(500)
        return UploadResult(500, terminal=False, retryable=True)

    result = upload_with_retries(
        CaptureFrame(capture.frame_path, capture.meta_path),
        _config(pending),
        post=always_retry,
        sleep=lambda _: None,
    )

    assert result.terminal is False
    assert result.retryable is True
    assert attempts == [500, 500, 500]
    assert capture.frame_path.exists()
    assert capture.meta_path.exists()


def test_iter_frames_skips_young_orphan_frame_without_metadata(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.meta_path.unlink()

    assert list(iter_frames(pending)) == []
    assert capture.frame_path.exists()


def test_iter_frames_deletes_old_orphan_frame_without_metadata(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.meta_path.unlink()
    old_time = 1_700_000_000
    os.utime(capture.frame_path, (old_time, old_time))

    assert list(iter_frames(pending, now=old_time + 1_000)) == []
    assert not capture.frame_path.exists()


def test_drain_once_drops_invalid_metadata_without_posting(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.meta_path.write_text("{not json", encoding="utf-8")

    def should_not_post(*_: object) -> UploadResult:
        raise AssertionError("invalid local metadata should not be posted")

    assert drain_once(_config(pending), post=should_not_post, sleep=lambda _: None) == 1
    assert not capture.frame_path.exists()
    assert not capture.meta_path.exists()


def test_validate_pending_frame_rejects_path_metadata_mismatch(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.metadata["session_uuid"] = "different-session"
    capture.meta_path.write_text(
        json.dumps(capture.metadata),
        encoding="utf-8",
    )

    error_message = validate_pending_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        pending,
    )

    assert error_message is not None
    assert "session_uuid" in error_message
    assert "does not match" in error_message


def test_validate_pending_frame_allows_local_when_requested_for_smoke_validation(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending, capture_storage_mode="local")

    upload_error = validate_pending_frame(CaptureFrame(capture.frame_path, capture.meta_path), pending)
    smoke_error = validate_pending_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        pending,
        allow_local=True,
    )

    assert upload_error == "metadata capture_storage_mode='local' is not uploadable"
    assert smoke_error is None


def test_validate_capture_tree_accepts_local_raw_present_payload(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    session_dir = pending / "cyberpunk-2077" / "present-session"
    session_dir.mkdir(parents=True)
    raw_path = session_dir / "present-0000000001.exr.raw"
    meta_path = session_dir / "present-0000000001.exr.json"
    raw_path.write_bytes(b"\x00" * 128)
    meta_path.write_text(
        json.dumps(
            {
                "capture_kind": "present_raw_backbuffer",
                "game_id": "cyberpunk-2077",
                "capture_storage_mode": "local",
                "session_uuid": "present-session",
                "frame_uuid": "present-0000000001",
                "captured_at_unix": 1_778_780_640.0,
                "sequence_index": 1,
                "width": 4,
                "height": 4,
                "dxgi_format": 10,
                "raw_path": str(raw_path),
            }
        ),
        encoding="utf-8",
    )

    issues = validate_capture_tree(pending, allow_local=True)

    assert issues == []


def test_validate_pending_frame_rejects_missing_required_hr_channels(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending, burst_tier="long", burst_index=3)
    capture.metadata["hr_source"] = "dlss-quality"
    capture.meta_path.write_text(json.dumps(capture.metadata), encoding="utf-8")

    error_message = validate_pending_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        pending,
    )

    assert error_message is not None
    assert "HR" in error_message


def test_validate_pending_frame_rejects_dimension_mismatch(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.metadata["hr_resolution"] = [999, 999]
    capture.meta_path.write_text(json.dumps(capture.metadata), encoding="utf-8")

    error_message = validate_pending_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        pending,
    )

    assert error_message is not None
    assert "hr_resolution" in error_message


def test_validate_pending_frame_rejects_nan_exr_channels(tmp_path: Path) -> None:
    pyexr = _require_pyexr()
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    width, height = capture.metadata["hr_resolution"]
    good_rgb = np.ones((height, width, 3), dtype=np.float32)
    lr = good_rgb.copy()
    lr[0, 0, 0] = np.nan
    pyexr.write(
        capture.frame_path,
        {
            "LR": lr,
            "HR": good_rgb,
            "Depth": np.ones((height, width, 1), dtype=np.float32),
            "Motion": np.zeros((height, width, 2), dtype=np.float32),
            "Normals": good_rgb,
        },
        channel_names={
            "LR": ["R", "G", "B"],
            "HR": ["R", "G", "B"],
            "Depth": ["Z"],
            "Motion": ["X", "Y"],
            "Normals": ["X", "Y", "Z"],
        },
        compression=pyexr.ZIP_COMPRESSION,
        compression_level=5,
    )

    error_message = validate_pending_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        pending,
    )

    assert error_message is not None
    assert "NaN or Inf" in error_message


def test_validate_capture_tree_reports_missing_sidecars_and_burst_gaps(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    burst_uuid = "11111111-1111-4111-8111-111111111111"
    make_synthetic_capture(
        pending,
        session_uuid="22222222-2222-4222-8222-222222222222",
        frame_uuid="33333333-3333-4333-8333-333333333330",
        burst_uuid=burst_uuid,
        burst_index=0,
        captured_at_unix=100,
    )
    make_synthetic_capture(
        pending,
        session_uuid="22222222-2222-4222-8222-222222222222",
        frame_uuid="33333333-3333-4333-8333-333333333332",
        burst_uuid=burst_uuid,
        burst_index=2,
        captured_at_unix=102,
    )
    orphan = make_synthetic_capture(
        pending,
        frame_uuid="44444444-4444-4444-8444-444444444444",
    )
    orphan.meta_path.unlink()

    issues = validate_capture_tree(pending)

    messages = [issue.message for issue in issues]
    assert "missing JSON sidecar for EXR" in messages
    assert any("indices are not contiguous" in message and "missing [1]" in message for message in messages)


def test_validate_capture_samples_script_reports_json(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    make_synthetic_capture(pending)

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "validate_capture_samples.py"),
            str(pending),
            "--json",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["issue_count"] == 0
    assert payload["storage_modes"] == {"invalid": 0, "local": 0, "missing": 0, "upload": 1}


def test_validate_capture_samples_script_can_scope_to_latest_session(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    stale = make_synthetic_capture(pending, session_uuid="stale-session")
    stale_meta = json.loads(stale.meta_path.read_text(encoding="utf-8"))
    stale_meta.pop("session_uuid")
    stale.meta_path.write_text(json.dumps(stale_meta), encoding="utf-8")

    newest = make_synthetic_capture(pending, session_uuid="newest-session")
    os.utime(newest.frame_path, (2000, 2000))
    os.utime(newest.meta_path, (2000, 2000))
    os.utime(stale.frame_path, (1000, 1000))
    os.utime(stale.meta_path, (1000, 1000))

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "validate_capture_samples.py"),
            str(pending),
            "--json",
            "--latest-session-count",
            "1",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["issue_count"] == 0
    assert payload["scoped_file_count"] == 2
    assert payload["validated_session_dirs"] == [str(newest.frame_path.parent)]


def test_validate_capture_samples_script_fails_empty_latest_scope(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "validate_capture_samples.py"),
            str(pending),
            "--json",
            "--latest-session-count",
            "1",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["issue_count"] == 1
    assert payload["issues"][0]["message"] == "no capture sessions found"


def test_enforce_pending_cap_deletes_oldest_pair(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    oldest = make_synthetic_capture(
        pending,
        session_uuid="z-session",
        frame_uuid="oldest",
        payload_bytes=256,
        captured_at_unix=100,
    )
    newest = make_synthetic_capture(
        pending,
        session_uuid="a-session",
        frame_uuid="newest",
        payload_bytes=256,
        captured_at_unix=200,
    )
    base_ns = 1_778_000_000_000_000_000
    os.utime(oldest.frame_path, ns=(base_ns, base_ns))
    os.utime(oldest.meta_path, ns=(base_ns, base_ns))
    os.utime(newest.frame_path, ns=(base_ns, base_ns))
    os.utime(newest.meta_path, ns=(base_ns, base_ns))
    newest_total = newest.frame_path.stat().st_size + newest.meta_path.stat().st_size

    deleted = enforce_pending_cap(pending, max_bytes=newest_total + 1)

    assert deleted == [oldest.frame_path]
    assert not oldest.frame_path.exists()
    assert not oldest.meta_path.exists()
    assert newest.frame_path.exists()
    assert newest.meta_path.exists()


def test_load_config_reads_windows_capture_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    pending = tmp_path / "pending"
    config_path.write_text(
        (
            "{"
            f'"pending_dir": "{pending}",'
            '"ingest_url": "https://example.test/ingest",'
            '"install_token": "abc",'
            '"max_pending_bytes": 123,'
            '"max_attempts": 2,'
            '"scan_interval_seconds": 9'
            "}"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.pending_dir == pending
    assert config.ingest_url == "https://example.test/ingest"
    assert config.install_token == "abc"
    assert config.max_pending_bytes == 123
    assert config.max_attempts == 2
    assert config.scan_interval_seconds == 9


def test_post_frame_returns_retryable_on_network_failure(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)

    result = post_frame(CaptureFrame(capture.frame_path, capture.meta_path), "http://127.0.0.1:1/ingest", "token", timeout=0.05)

    assert result.status_code is None
    assert result.retryable is True
    assert result.terminal is False


def test_post_frame_includes_server_validation_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)

    def bad_request(req, timeout):  # noqa: ANN001 - matches urllib hook.
        headers = Message()
        body = b'{"detail":{"error":"frame_filename_mismatch","frame_uuid":"abc"}}'
        raise error.HTTPError(req.full_url, 400, "Bad Request", headers, io.BytesIO(body))

    monkeypatch.setattr(uploader.request, "urlopen", bad_request)

    result = post_frame(
        CaptureFrame(capture.frame_path, capture.meta_path),
        "https://example.test/ingest",
        "token",
    )

    assert result.status_code == 400
    assert result.terminal is True
    assert "frame_filename_mismatch" in result.message

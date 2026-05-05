from __future__ import annotations

from pathlib import Path

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
)
from tests.capture.test_fixtures import make_synthetic_capture


def _config(pending: Path) -> UploadConfig:
    return UploadConfig(
        pending_dir=pending,
        ingest_url="http://127.0.0.1:1/ingest",
        install_token="token",
        max_attempts=3,
        backoff_seconds=(0.0, 0.0, 0.0),
    )


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


def test_upload_with_retries_drops_after_exhausted_5xx(tmp_path: Path) -> None:
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

    assert result.terminal is True
    assert attempts == [500, 500, 500]
    assert not capture.frame_path.exists()
    assert not capture.meta_path.exists()


def test_drain_once_deletes_orphan_frame_without_metadata(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    capture = make_synthetic_capture(pending)
    capture.meta_path.unlink()

    assert list(iter_frames(pending)) == []
    assert not capture.frame_path.exists()
    assert drain_once(_config(pending), post=lambda *_: UploadResult(200, True, False), sleep=lambda _: None) == 0


def test_enforce_pending_cap_deletes_oldest_pair(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    oldest = make_synthetic_capture(pending, frame_uuid="oldest", payload_bytes=256)
    newest = make_synthetic_capture(pending, frame_uuid="newest", payload_bytes=256)
    oldest.frame_path.touch()
    oldest.meta_path.touch()
    newest.frame_path.touch()
    newest.meta_path.touch()
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

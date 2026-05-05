from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path

from oss.capture.uploader import UploadConfig, drain_once
from tests.capture.test_fixtures import make_synthetic_capture


class ScriptedIngestHandler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    requests_seen: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("Content-Length", "0"))
        ScriptedIngestHandler.requests_seen.append(self.rfile.read(length))
        status = ScriptedIngestHandler.statuses.pop(0) if ScriptedIngestHandler.statuses else 200
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_server(statuses: list[int]) -> tuple[ThreadingHTTPServer, str]:
    ScriptedIngestHandler.statuses = list(statuses)
    ScriptedIngestHandler.requests_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedIngestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/ingest"


def test_uploader_fake_server_roundtrip_deletes_terminal_and_exhausted_frames(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    burst_uuid = "11111111-1111-4111-8111-111111111111"
    ok = make_synthetic_capture(pending, session_uuid="session", frame_uuid="000-0-ok", burst_uuid=burst_uuid, burst_index=0)
    trickle_static = make_synthetic_capture(
        pending,
        session_uuid="session",
        frame_uuid="000-a-trickle-static",
        capture_mode="trickle",
        burst_tier=None,
    )
    trickle_pair_tplus = make_synthetic_capture(
        pending,
        session_uuid="session",
        frame_uuid="000-b-trickle-pair-tplus",
        capture_mode="trickle",
        burst_uuid="33333333-3333-4333-8333-333333333333",
        burst_index=1,
        burst_tier="short",
    )
    long = make_synthetic_capture(
        pending,
        session_uuid="session",
        frame_uuid="001-long",
        burst_uuid="22222222-2222-4222-8222-222222222222",
        burst_index=12,
        burst_tier="long",
    )
    regular = make_synthetic_capture(pending, session_uuid="session", frame_uuid="002-regular", capture_mode="regular")
    insane = make_synthetic_capture(pending, session_uuid="session", frame_uuid="003-insane", capture_mode="INSANE")
    rejected = make_synthetic_capture(pending, session_uuid="session", frame_uuid="004-rejected", burst_uuid=burst_uuid, burst_index=1)
    exhausted = make_synthetic_capture(pending, session_uuid="session", frame_uuid="005-exhausted", burst_uuid=burst_uuid, burst_index=2)
    server, ingest_url = _start_server([200, 200, 200, 200, 200, 200, 400, 500, 500])
    try:
        config = UploadConfig(
            pending_dir=pending,
            ingest_url=ingest_url,
            install_token="test-token",
            max_attempts=2,
            backoff_seconds=(0.0, 0.0),
        )
        assert drain_once(config, sleep=lambda _: None) == 8
    finally:
        server.shutdown()
        server.server_close()

    for capture in (ok, trickle_static, trickle_pair_tplus, long, regular, insane, rejected, exhausted):
        assert not capture.frame_path.exists()
        assert not capture.meta_path.exists()

    assert len(ScriptedIngestHandler.requests_seen) == 9
    first_body = ScriptedIngestHandler.requests_seen[0]
    assert b'name="frame"; filename="000-0-ok.exr"' in first_body
    assert b'name="meta"; filename="000-0-ok.json"' in first_body
    assert b'"schema_version": 1' in first_body
    assert b'"burst_uuid": "11111111-1111-4111-8111-111111111111"' in first_body
    assert b'"burst_index": 0' in first_body
    trickle_static_body = ScriptedIngestHandler.requests_seen[1]
    assert b'name="frame"; filename="000-a-trickle-static.exr"' in trickle_static_body
    assert b'"capture_mode": "trickle"' in trickle_static_body
    assert b'"burst_uuid"' not in trickle_static_body
    trickle_pair_body = ScriptedIngestHandler.requests_seen[2]
    assert b'name="frame"; filename="000-b-trickle-pair-tplus.exr"' in trickle_pair_body
    assert b'"capture_mode": "trickle"' in trickle_pair_body
    assert b'"burst_tier": "short"' in trickle_pair_body
    assert b'"burst_index": 1' in trickle_pair_body
    assert b'"hr_source": "none"' in trickle_pair_body
    long_body = ScriptedIngestHandler.requests_seen[3]
    assert b'name="frame"; filename="001-long.exr"' in long_body
    assert b'"burst_tier": "long"' in long_body
    assert b'"hr_source": "none"' in long_body
    assert b'"burst_index": 12' in long_body
    regular_body = ScriptedIngestHandler.requests_seen[4]
    assert b'"capture_mode": "regular"' in regular_body
    insane_body = ScriptedIngestHandler.requests_seen[5]
    assert b'"capture_mode": "INSANE"' in insane_body


def test_pending_cap_evicts_oldest_pairs_before_upload(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    oldest = make_synthetic_capture(pending, frame_uuid="000-oldest", payload_bytes=220, captured_at_unix=1)
    newest = make_synthetic_capture(pending, frame_uuid="001-newest", payload_bytes=220, captured_at_unix=2)
    os.utime(oldest.frame_path, (1, 1))
    os.utime(oldest.meta_path, (1, 1))
    os.utime(newest.frame_path, (2, 2))
    os.utime(newest.meta_path, (2, 2))
    newest_total = newest.frame_path.stat().st_size + newest.meta_path.stat().st_size
    server, ingest_url = _start_server([200])
    try:
        config = UploadConfig(
            pending_dir=pending,
            ingest_url=ingest_url,
            install_token="test-token",
            max_pending_bytes=newest_total + 1,
            max_attempts=1,
            backoff_seconds=(0.0,),
        )
        assert drain_once(config, sleep=lambda _: None) == 1
    finally:
        server.shutdown()
        server.server_close()

    assert not oldest.frame_path.exists()
    assert not oldest.meta_path.exists()
    assert not newest.frame_path.exists()
    assert not newest.meta_path.exists()
    assert len(ScriptedIngestHandler.requests_seen) == 1

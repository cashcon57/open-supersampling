from __future__ import annotations

import os
import time
from pathlib import Path

import scripts.training_dashboard as dashboard
from scripts.training_dashboard import DashboardHandler


def _handler_for(tmp_path: Path):
    class Handler(DashboardHandler):
        def _send_json(self, payload, status: int = 200) -> None:
            self.sent_status = status
            self.sent_payload = payload

        def _send_text(self, text: str, status: int = 200) -> None:
            self.sent_status = status
            self.sent_payload = text

    Handler.output_dir = tmp_path.resolve()
    Handler.log_file = (tmp_path / "train.log").resolve()
    Handler.log_file.write_text("")
    return Handler


def _get_json(handler_cls, path: str) -> tuple[int, dict]:
    handler = object.__new__(handler_cls)
    handler.path = path
    handler.do_GET()
    return handler.sent_status, handler.sent_payload


def _write_log(path: Path, body: str, mtime: float) -> None:
    path.write_text(body)
    os.utime(path, (mtime, mtime))


def test_codex_log_stream_endpoint_merges_multiple_active_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    now = time.time()
    first = log_dir / "codex-first.log"
    second = log_dir / "codex-second.log"
    _write_log(first, "codex\nfirst reasoning\n", now - 20)
    _write_log(second, "codex\nsecond reasoning\n", now - 5)
    monkeypatch.setattr(dashboard, "CODEX_LOG_DIR", log_dir)

    status, payload = _get_json(
        _handler_for(tmp_path),
        "/api/codex-log-stream?files=codex-first.log,codex-second.log",
    )

    assert status == 200
    assert payload["mode"] == "stream"
    assert payload["files"] == ["codex-first.log", "codex-second.log"]
    assert payload["entries"] == 2
    html = payload["html"]
    assert "codex-first.log" in html
    assert "codex-second.log" in html
    assert "first reasoning" in html
    assert "second reasoning" in html
    assert html.index("first reasoning") < html.index("second reasoning")

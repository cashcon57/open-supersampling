from __future__ import annotations

import os
from pathlib import Path

from scripts.training_dashboard import DashboardHandler


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    os.utime(path, (mtime, mtime))


def _make_run(parent: Path, name: str, mtime: float, *, viz: bool = False, score: bool = False) -> Path:
    run = parent / name
    run.mkdir(parents=True)
    _touch(run / "metrics.json", mtime)
    if score:
        _touch(run / "score_log.json", mtime + 1)
    if viz:
        _touch(run / "viz" / "step-00000001.png", mtime + 2)
    os.utime(run, (mtime, mtime))
    return run


def _handler_for(output_dir: Path, log_file: Path):
    class Handler(DashboardHandler):
        def _send_json(self, payload, status: int = 200) -> None:
            self.sent_status = status
            self.sent_payload = payload

        def _send_text(self, text: str, status: int = 200) -> None:
            self.sent_status = status
            self.sent_payload = text

    Handler.output_dir = output_dir.resolve()
    Handler.log_file = log_file.resolve()
    return Handler


def _get_json(handler_cls, path: str) -> tuple[int, dict]:
    handler = object.__new__(handler_cls)
    handler.path = path
    handler.do_GET()
    return handler.sent_status, handler.sent_payload


def test_runs_endpoint_returns_expected_json_shape(tmp_path: Path) -> None:
    parent = tmp_path / "checkpoints"
    old = _make_run(parent, "srcnn-v5-pixel-temporal", 1000, score=True)
    _make_run(parent, "srcnn-v6-alpha", 2000, viz=True)
    _make_run(parent, "not-a-training-run", 3000)
    log_file = tmp_path / "train.log"
    log_file.write_text("")

    status, payload = _get_json(_handler_for(old, log_file), "/api/runs")

    assert status == 200
    assert payload["default_run"] == "srcnn-v6-alpha"
    assert [r["name"] for r in payload["runs"]] == [
        "srcnn-v6-alpha",
        "srcnn-v5-pixel-temporal",
    ]
    first = payload["runs"][0]
    assert set(first) == {
        "name",
        "path",
        "last_modified",
        "has_train_log",
        "has_viz",
        "has_score_log",
    }
    assert first["has_train_log"] is True
    assert first["has_viz"] is True
    assert first["has_score_log"] is False


def test_run_query_denies_path_traversal(tmp_path: Path) -> None:
    parent = tmp_path / "checkpoints"
    run = _make_run(parent, "srcnn-v5-pixel-temporal", 1000)
    log_file = tmp_path / "train.log"
    log_file.write_text("")

    status, payload = _get_json(_handler_for(run, log_file), "/api/info?run=../../etc")

    assert status == 403
    assert payload["error"] == "denied run selector"


def test_info_defaults_to_most_recent_run(tmp_path: Path) -> None:
    parent = tmp_path / "checkpoints"
    old = _make_run(parent, "srcnn-v5-pixel-temporal", 1000)
    new = _make_run(parent, "srcnn-v6-alpha", 5000)
    log_file = tmp_path / "train.log"
    log_file.write_text("")

    status, payload = _get_json(_handler_for(old, log_file), "/api/info")

    assert status == 200
    assert payload["run"] == new.name
    assert payload["output_dir"] == str(new.resolve())

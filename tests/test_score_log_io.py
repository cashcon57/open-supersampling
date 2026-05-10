from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from scripts._score_log_io import append_score_log_row, write_score_log_rows
from scripts.sr_v6_held_out import _update_score_log


def test_append_score_log_row_replaces_existing_step(tmp_path: Path) -> None:
    path = tmp_path / "score_log.json"

    append_score_log_row(path, {"step": 2, "value": "old"})
    append_score_log_row(path, {"step": 2, "value": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == [{"step": 2, "value": "new"}]


def test_append_score_log_row_is_thread_safe_and_orders_steps(tmp_path: Path) -> None:
    path = tmp_path / "score_log.json"
    barrier = Barrier(5)

    def append(step: int) -> None:
        barrier.wait()
        append_score_log_row(path, {"step": step, "value": f"row-{step}"})

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(append, [4, 2, 0, 3, 1]))

    rows = json.loads(path.read_text(encoding="utf-8"))
    assert [row["step"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["value"] for row in rows] == ["row-0", "row-1", "row-2", "row-3", "row-4"]


def test_sr_v6_held_out_append_score_log_is_thread_safe(tmp_path: Path) -> None:
    path = tmp_path / "score_log.json"
    barrier = Barrier(5)

    def append(step: int) -> None:
        barrier.wait()
        _update_score_log(path, {"step": step, "value": f"held-out-{step}"})

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(append, [4, 2, 0, 3, 1]))

    rows = json.loads(path.read_text(encoding="utf-8"))
    assert [row["step"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["value"] for row in rows] == [
        "held-out-0",
        "held-out-1",
        "held-out-2",
        "held-out-3",
        "held-out-4",
    ]


def test_write_score_log_rows_preserves_existing_step_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "score_log.json"
    append_score_log_row(path, {"step": 10, "value": "fresh-heldout"})

    write_score_log_rows(
        path,
        [
            {"step": 5, "value": "trainer-known"},
            {"step": 10, "value": "stale-trainer-copy"},
        ],
    )

    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows == [
        {"step": 5, "value": "trainer-known"},
        {"step": 10, "value": "fresh-heldout"},
    ]

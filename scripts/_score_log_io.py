"""Shared IO helpers for dashboard ``score_log.json`` files."""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

if os.name == "nt":  # pragma: no cover - CI/dev paths are Unix-like.
    import msvcrt
else:  # pragma: no cover - exercised indirectly by public APIs.
    import fcntl


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _thread_lock_for(lock_path: Path) -> threading.Lock:
    key = lock_path.resolve(strict=False)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def _lock_fd(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(path)
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if os.name == "nt" and os.path.getsize(lock_path) == 0:  # pragma: no cover
                os.write(fd, b"\0")
            _lock_fd(fd)
            try:
                yield
            finally:
                _unlock_fd(fd)
        finally:
            os.close(fd)


def _read_rows_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _write_rows_unlocked(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = _tmp_path(path)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump([dict(row) for row in rows], f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_score_log_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Merge ``rows`` into ``path`` using the shared sidecar lock.

    Legacy trainers keep an in-memory score_log list that may be stale while
    the held-out evaluator is appending real dashboard rows. Preserve existing
    rows on duplicate steps so those trainer flushes cannot clobber newer
    evaluator output.
    """
    with _locked(path):
        merged: dict[Any, dict[str, Any]] = {}
        ordered_without_step: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            if "step" in row_dict:
                merged[row_dict.get("step")] = row_dict
            else:
                ordered_without_step.append(row_dict)
        for existing in _read_rows_unlocked(path):
            if "step" in existing:
                merged[existing.get("step")] = existing
            else:
                ordered_without_step.append(existing)
        merged_rows = ordered_without_step + sorted(
            merged.values(),
            key=lambda existing: int(existing.get("step", -1)),
        )
        _write_rows_unlocked(path, merged_rows)


def append_score_log_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append or replace one score row, deduping by matching ``step``."""
    row_dict = dict(row)
    with _locked(path):
        rows = _read_rows_unlocked(path)
        rows = [existing for existing in rows if existing.get("step") != row_dict.get("step")]
        rows.append(row_dict)
        rows.sort(key=lambda existing: int(existing.get("step", -1)))
        _write_rows_unlocked(path, rows)

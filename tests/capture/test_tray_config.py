"""Tests for the tray-app config + storage layers.

These tests are platform-independent (no Win32 calls); they verify config
round-trip, drive-picker logic against a fake candidates list, and
janitor cleanup against a temp dir. The Windows-specific bits (DLL
injection, process watcher, Steam library scanner) get their own tests
when those modules land.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from oss.capture.tray import config as cfg_mod
from oss.capture.tray import storage


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_config_load_returns_defaults_when_no_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: tmp_path)
    cfg = cfg_mod.load()
    assert cfg.capture_mode == "lite"
    assert cfg.output_drive_override is None
    assert cfg.paused is False
    assert cfg.enabled_games == {}
    assert cfg.disk_cap_bytes == cfg_mod.DEFAULT_DISK_CAP_BYTES


def test_config_save_then_load_roundtrips(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: tmp_path)
    written = cfg_mod.TrayConfig(
        capture_mode="INSANE",
        output_drive_override="G:\\",
        paused=True,
        enabled_games={"cyberpunk-2077": True, "alan-wake-2": False},
        disk_cap_bytes=50 * 1024**3,
    )
    cfg_mod.save(written)
    read_back = cfg_mod.load()
    assert read_back == written


def test_config_corrupt_file_falls_back_to_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: tmp_path)
    (tmp_path / "tray-config.json").write_text("{not valid json", encoding="utf-8")
    cfg = cfg_mod.load()
    assert cfg.capture_mode == "lite"  # defaults restored


def test_config_unknown_keys_are_dropped(monkeypatch, tmp_path) -> None:
    """Forward-compat: a config from a future schema_version that adds
    fields the current code does not know about must load without crashing.
    """
    monkeypatch.setattr(cfg_mod, "_config_dir", lambda: tmp_path)
    (tmp_path / "tray-config.json").write_text(
        json.dumps({
            "capture_mode": "regular",
            "schema_version": 99,
            "future_field_we_dont_know_yet": [1, 2, 3],
        }),
        encoding="utf-8",
    )
    cfg = cfg_mod.load()
    assert cfg.capture_mode == "regular"


def test_config_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="capture_mode"):
        cfg_mod.TrayConfig(capture_mode="ULTRA-DUPER-MODE")


def test_config_too_small_disk_cap_raises() -> None:
    with pytest.raises(ValueError, match="disk_cap_bytes"):
        cfg_mod.TrayConfig(disk_cap_bytes=512 * 1024 * 1024)  # 512 MiB


# ---------------------------------------------------------------------------
# Drive picker
# ---------------------------------------------------------------------------


def test_pick_output_drive_honors_override() -> None:
    # Override wins regardless of whether the candidates exist.
    assert storage.pick_output_drive(override="X:\\", candidates=()) == "X:\\"


def test_pick_output_drive_returns_none_when_no_candidates(monkeypatch) -> None:
    # On non-Windows, _drive_exists always returns False, so the candidate
    # list is filtered to empty and pick_output_drive returns None.
    assert storage.pick_output_drive(candidates=("Z:\\",)) is None


# ---------------------------------------------------------------------------
# Janitor cleanup
# ---------------------------------------------------------------------------


def test_cleanup_to_cap_no_op_when_under_cap(tmp_path: Path) -> None:
    base = tmp_path / "oss-captures"
    base.mkdir()
    (base / "a.exr").write_bytes(b"x" * 1000)
    (base / "b.exr").write_bytes(b"x" * 1000)

    deleted = storage.cleanup_to_cap(str(tmp_path), cap_bytes=10_000)
    assert deleted == 0
    assert (base / "a.exr").exists()
    assert (base / "b.exr").exists()


def test_cleanup_to_cap_deletes_oldest_first(tmp_path: Path) -> None:
    base = tmp_path / "oss-captures"
    base.mkdir()
    paths = [base / f"frame-{i}.exr" for i in range(5)]
    for i, p in enumerate(paths):
        p.write_bytes(b"x" * 1000)
        # Stamp older mtimes on lower-indexed files so the cleanup deletes
        # them first, preserving the youngest.
        os.utime(p, (time.time() - (5 - i) * 60, time.time() - (5 - i) * 60))

    # Cap allows only 2000 bytes (= 2 files); 3 oldest must be deleted.
    deleted = storage.cleanup_to_cap(str(tmp_path), cap_bytes=2000)
    assert deleted == 3
    assert not paths[0].exists()
    assert not paths[1].exists()
    assert not paths[2].exists()
    assert paths[3].exists()
    assert paths[4].exists()


def test_total_captures_bytes_walks_subdirs(tmp_path: Path) -> None:
    base = tmp_path / "oss-captures"
    (base / "cyberpunk-2077" / "session-1").mkdir(parents=True)
    (base / "cyberpunk-2077" / "session-1" / "frame.exr").write_bytes(b"x" * 4096)
    (base / "alan-wake-2" / "session-1").mkdir(parents=True)
    (base / "alan-wake-2" / "session-1" / "frame.exr").write_bytes(b"x" * 8192)

    assert storage.total_captures_bytes(str(tmp_path)) == 4096 + 8192

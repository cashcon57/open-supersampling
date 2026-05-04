"""Tests for TrajectoryWindowDataset (v5 gaussian-temporal Task 9)."""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal import (
    TrajectoryWindowDataset,
    default_collate_window,
)


class _FakeBase:
    """Minimal stub: 5 frames in seq A, 3 frames in seq B (8 total)."""

    def __init__(self) -> None:
        self._seq_keys = ["A"] * 5 + ["B"] * 3

    def __len__(self) -> int:
        return len(self._seq_keys)

    def trajectory_key(self, idx: int) -> str:
        return self._seq_keys[idx]

    def __getitem__(self, idx: int):
        return {
            "lr_frame": torch.full((3, 8, 8), float(idx)),
            "depth": torch.full((1, 8, 8), float(idx)),
            "motion": torch.full((2, 8, 8), float(idx)),
            "normals": torch.full((3, 8, 8), float(idx)),
            "canvas_hint": torch.full((3, 8, 8), float(idx)),
            "gt_hr_frame": torch.full((3, 16, 16), float(idx)),
        }


def test_window_count_respects_trajectory_boundary() -> None:
    base = _FakeBase()
    ds = TrajectoryWindowDataset(base, window=5)
    # Only seq A (5 frames) supports a 5-frame window; seq B (3 frames) cannot.
    assert len(ds) == 1


def test_window_frames_share_trajectory_key() -> None:
    base = _FakeBase()
    ds = TrajectoryWindowDataset(base, window=5)
    sample = ds[0]
    assert "frames" in sample
    assert "trajectory_key" in sample
    assert len(sample["frames"]) == 5
    # Frames are consecutive indices 0..4 in seq A; lr value reflects index.
    for k, frame in enumerate(sample["frames"]):
        assert frame["lr_frame"][0, 0, 0].item() == float(k)
    # All frames originate from the same trajectory.
    assert sample["trajectory_key"] == "A"


def test_window_smaller_window_returns_more_windows() -> None:
    base = _FakeBase()
    ds = TrajectoryWindowDataset(base, window=3)
    # Seq A (5 frames): windows starting at 0,1,2 = 3 windows.
    # Seq B (3 frames): window starting at 5 = 1 window.
    assert len(ds) == 4


def test_default_collate_window_stacks_per_field() -> None:
    base = _FakeBase()
    ds = TrajectoryWindowDataset(base, window=5)
    sample = ds[0]
    # Build a batch of size 2 by duplicating the same window.
    batch = default_collate_window([sample, sample])
    assert "frames" in batch
    assert len(batch["frames"]) == 5
    for i in range(5):
        assert batch["frames"][i]["lr_frame"].shape == (2, 3, 8, 8)
        assert batch["frames"][i]["depth"].shape == (2, 1, 8, 8)
        assert batch["frames"][i]["motion"].shape == (2, 2, 8, 8)
        assert batch["frames"][i]["normals"].shape == (2, 3, 8, 8)
        assert batch["frames"][i]["canvas_hint"].shape == (2, 3, 8, 8)
        assert batch["frames"][i]["gt_hr_frame"].shape == (2, 3, 16, 16)
    assert batch["trajectory_key"] == ["A", "A"]


def test_window_requires_trajectory_key() -> None:
    class _NoKey:
        def __len__(self) -> int:
            return 5

        def __getitem__(self, idx: int):
            return {"lr_frame": torch.zeros(3, 8, 8)}

    try:
        TrajectoryWindowDataset(_NoKey(), window=5)
    except TypeError as e:
        assert "trajectory_key" in str(e)
    else:
        raise AssertionError("expected TypeError when base has no trajectory_key")

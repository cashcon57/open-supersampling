"""Tests for SequentialPairDataset."""
from __future__ import annotations

import torch

from oss.sr.temporal import SequentialPairDataset, default_collate_pair


class _FakeBase:
    """Minimal stub: 5 frames in seq A, 3 frames in seq B."""

    def __init__(self) -> None:
        self.scale = 2.0
        self._seq_keys = ["A"] * 5 + ["B"] * 3

    def __len__(self) -> int:
        return len(self._seq_keys)

    def trajectory_key(self, idx: int) -> str:
        return self._seq_keys[idx]

    def __getitem__(self, idx: int):
        k = self._seq_keys[idx]
        # Tiny tensors; spatial sizes match (LR=8, HR=16) and 12-ch contract.
        return {
            "lr_frame": torch.full((3, 8, 8), float(idx)),
            "depth": torch.full((1, 8, 8), float(idx)),
            "motion": torch.full((2, 8, 8), float(idx)),
            "normals": torch.full((3, 8, 8), float(idx)),
            "canvas_hint": torch.full((3, 8, 8), float(idx)),
            "gt_hr_frame": torch.full((3, 16, 16), float(idx)),
            "_seq": k,
        }


def test_pair_count_excludes_boundaries() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    # 4 pairs in A (idx 0..3) + 2 pairs in B (idx 5..6) = 6 valid pairs.
    assert len(ds) == 6


def test_pair_returns_consecutive() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    pair = ds[0]
    assert pair["t"]["lr_frame"][0, 0, 0].item() == 0.0
    assert pair["t_plus_1"]["lr_frame"][0, 0, 0].item() == 1.0
    assert pair["is_first_in_seq"] is True


def test_pair_in_middle_is_not_first() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    # Find a pair where idx_t > 0 within its seq.
    pair = ds[1]
    assert pair["is_first_in_seq"] is False


def test_collate_pair() -> None:
    base = _FakeBase()
    ds = SequentialPairDataset(base)
    batch = default_collate_pair([ds[0], ds[1], ds[2]])
    assert batch["t_lr"].shape == (3, 3, 8, 8)
    assert batch["tp1_lr"].shape == (3, 3, 8, 8)
    assert batch["t_gt_hr"].shape == (3, 3, 16, 16)
    assert batch["is_first_in_seq"].shape == (3,)

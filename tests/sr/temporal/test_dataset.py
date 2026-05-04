"""Tests for SequentialPairDataset."""
from __future__ import annotations

import torch

from oss.gaussian.data import GaussianTrainingExample
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


def test_collate_pair_accepts_gaussian_training_examples() -> None:
    def _example(idx: int) -> GaussianTrainingExample:
        return GaussianTrainingExample(
            lr_frame=torch.full((3, 8, 8), float(idx)),
            depth=torch.full((1, 8, 8), float(idx)),
            motion=torch.full((2, 8, 8), float(idx)),
            canvas_hint=torch.full((3, 8, 8), float(idx)),
            gt_hr_frame=torch.full((3, 16, 16), float(idx)),
            normals=None,
        )

    pair = {
        "t": _example(0),
        "t_plus_1": _example(1),
        "is_first_in_seq": True,
    }
    batch = default_collate_pair([pair])
    assert batch["t_lr"].shape == (1, 3, 8, 8)
    assert batch["tp1_motion"].shape == (1, 2, 8, 8)
    assert batch["t_normals"].shape == (1, 3, 8, 8)
    assert torch.equal(batch["t_normals"], torch.zeros_like(batch["t_normals"]))


def test_pair_stride_two() -> None:
    """Codex MEDIUM finding: API gap on pair_stride. Default=1; larger
    strides skip ``pair_stride-1`` intermediate frames per pair, and pairs
    that would cross trajectory boundaries are excluded."""
    base = _FakeBase()
    ds = SequentialPairDataset(base, pair_stride=2)
    # Seq A (5 frames): pairs (0,2), (1,3), (2,4) = 3.
    # Seq B (3 frames): pairs (5,7) = 1.
    # Total = 4.
    assert len(ds) == 4
    pair = ds[0]
    assert pair["t"]["lr_frame"][0, 0, 0].item() == 0.0
    assert pair["t_plus_1"]["lr_frame"][0, 0, 0].item() == 2.0


def test_pair_stride_invalid_raises() -> None:
    base = _FakeBase()
    try:
        SequentialPairDataset(base, pair_stride=0)
    except ValueError as e:
        assert "pair_stride" in str(e)
    else:
        raise AssertionError("expected ValueError on pair_stride=0")

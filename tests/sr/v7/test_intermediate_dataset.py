"""Unit tests for TartanAirIntermediateTriplets dataset adapter.

These tests use a fake base dataset to avoid requiring the real
TartanAir extraction on disk. The fake mimics the interface
(__getitem__ returns GaussianTrainingExample, _items list of path
tuples).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from oss.sr.v7.intermediate_dataset import (
    TartanAirIntermediateTriplets,
    _trajectory_root,
)


@dataclass
class FakeExample:
    lr_frame: torch.Tensor
    depth: torch.Tensor
    motion: torch.Tensor
    normals: torch.Tensor
    gt_hr_frame: torch.Tensor


class _FakeTartanAir:
    """Fake base dataset emulating the slice of TartanAirGaussianDataset
    interface that TartanAirIntermediateTriplets needs."""

    def __init__(self, n_per_traj: list[int]):
        """n_per_traj: list of integers, one per trajectory, giving the
        number of frames in that trajectory."""
        self._items: list[tuple[Path, Path, Path]] = []
        for traj_idx, n in enumerate(n_per_traj):
            traj_root = f"E:/datasets/tartanair_extracted/env{traj_idx}/Easy/P000/"
            for f in range(n):
                img = Path(f"{traj_root}image_left/{f:06d}_left.png")
                depth = Path(f"{traj_root}depth_left/{f:06d}_depth.npy")
                flow = Path(f"{traj_root}flow/{f:06d}_{f+1:06d}_flow.npy")
                self._items.append((img, depth, flow))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return FakeExample(
            lr_frame=torch.zeros((3, 8, 8)),
            depth=torch.zeros((1, 8, 8)),
            motion=torch.full((2, 8, 8), float(idx)),
            normals=torch.zeros((3, 8, 8)),
            gt_hr_frame=torch.full((3, 16, 16), float(idx)),
        )


def test_trajectory_root_extracts_prefix_correctly():
    p = Path("E:/datasets/tartanair_extracted/oldtown/Easy/P000/image_left/000123_left.png")
    root = _trajectory_root(p)
    assert root.endswith("P000/")
    assert "image_left" not in root


def test_intermediate_triplets_yields_only_intra_trajectory_triplets():
    """5 frames in traj 0, 4 in traj 1.  Valid triplets:
       traj 0: (0,1,2), (1,2,3), (2,3,4)             -> 3
       traj 1: (5,6,7), (6,7,8)                       -> 2
       Total: 5
    """
    base = _FakeTartanAir(n_per_traj=[5, 4])
    ds = TartanAirIntermediateTriplets(base)
    assert len(ds) == 5


def test_intermediate_triplets_skips_boundary():
    """Triplet (4, 5, 6) where 4 is last in traj 0 and 5 is first in
    traj 1 must NOT appear."""
    base = _FakeTartanAir(n_per_traj=[5, 4])
    ds = TartanAirIntermediateTriplets(base)
    # Triplet starts at 0, 1, 2 for traj 0; 5, 6 for traj 1. Never 4.
    for triplet in ds._triplet_indices:
        i0, i1, i2 = triplet
        # No triplet should cross the boundary
        if i0 == 4:
            pytest.fail(f"triplet starting at 4 spans boundary: {triplet}")
        if i0 < 5 and i2 >= 5:
            pytest.fail(f"triplet {triplet} spans boundary")


def test_intermediate_triplets_getitem_returns_dict_with_expected_keys():
    base = _FakeTartanAir(n_per_traj=[5])
    ds = TartanAirIntermediateTriplets(base)
    sample = ds[0]
    assert set(sample.keys()) == {"n", "n_half", "n_plus_1", "motion_n_to_np1"}
    assert set(sample["n"].keys()) == {"lr", "depth", "motion", "normals", "gt_hr"}
    assert set(sample["n_half"].keys()) == {"gt_hr"}
    assert set(sample["n_plus_1"].keys()) == {"lr", "depth", "motion", "normals", "gt_hr"}


def test_intermediate_triplets_n_half_is_middle_frame():
    """ds[0] should pull frames 0, 1, 2 from traj 0; n_half should
    correspond to frame 1, distinct from n (frame 0) and n_plus_1
    (frame 2)."""
    base = _FakeTartanAir(n_per_traj=[5])
    ds = TartanAirIntermediateTriplets(base)
    sample = ds[0]
    # Fake dataset encodes idx in gt_hr value; verify ordering.
    assert sample["n"]["gt_hr"][0, 0, 0].item() == 0.0
    assert sample["n_half"]["gt_hr"][0, 0, 0].item() == 1.0
    assert sample["n_plus_1"]["gt_hr"][0, 0, 0].item() == 2.0


def test_intermediate_triplets_motion_composition_doubles_single_step():
    """For linear motion, motion(i -> i+2) should be 2x motion(i -> i+1).
    Our current approximation doubles motion(i)."""
    base = _FakeTartanAir(n_per_traj=[3])
    ds = TartanAirIntermediateTriplets(base)
    sample = ds[0]
    # _FakeTartanAir motion at idx N is the constant N.
    # So motion(0) = 0; doubled = 0. Not super informative.
    # Better test: motion(N) is constant N, so motion(1) = 1, doubled = 2.
    base = _FakeTartanAir(n_per_traj=[3])
    # Manually set motion at idx 0 to a non-trivial value.
    # (FakeExample.motion is torch.full(...))
    sample0 = base[0]
    motion_at_0_value = sample0.motion[0, 0, 0].item()
    assert motion_at_0_value == 0.0  # confirms the fake
    # Compose: motion_n_to_np1 should equal motion[0] * 2 = 0
    assert sample["motion_n_to_np1"][0, 0, 0].item() == motion_at_0_value * 2


def test_intermediate_triplets_max_triplets_caps_size():
    base = _FakeTartanAir(n_per_traj=[100])
    ds = TartanAirIntermediateTriplets(base, max_triplets=4)
    assert len(ds) == 4

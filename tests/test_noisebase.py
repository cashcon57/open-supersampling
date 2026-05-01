"""Tests for ``ors.data.noisebase.NoiseBaseDataset``.

The synthetic fixture test must work without any real NoiseBase data on
disk: it builds a tiny NoiseBase-format Zarr ZipStore in ``tmp_path`` and
runs the loader against it. The ``test_noisebase_real_data`` test is
skipped unless ``ORS_NOISEBASE_ROOT`` is set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from oss.data.noisebase import NoiseBaseDataset, _write_synthetic_sequence


# ---------------------------------------------------------------------------
# Synthetic-fixture test (always runs)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Two synthetic 4-frame NoiseBase sequences at 16x16, 2 spp."""
    for i in range(2):
        _write_synthetic_sequence(
            tmp_path / f"scene{i:04d}.zip",
            frames=4,
            height=16,
            width=16,
            samples=2,
            seed=i,
        )
    return tmp_path


def test_noisebase_dataset_shapes(synthetic_root: Path) -> None:
    seq_len = 3
    hr_h, hr_w = 32, 32     # upsample fixture so HR > on-disk
    scale = 2.0
    lr_h, lr_w = int(hr_h / scale), int(hr_w / scale)

    ds = NoiseBaseDataset(
        root=synthetic_root,
        sequence_length=seq_len,
        resolution=(hr_h, hr_w),
        scale_factor=scale,
        split="train",
    )

    # Tiny synthetic dataset (n=2 < 5) → split='train' returns all sequences.
    assert len(ds) == 2

    item = ds[0]
    expected_keys = {"color_lr", "gt_hr", "motion_lr", "depth_lr", "normals_lr", "albedo_lr"}
    assert set(item.keys()) == expected_keys

    # All tensors must be float32 and 4D (T, C, H, W).
    for k, v in item.items():
        assert isinstance(v, torch.Tensor), f"{k} is {type(v)}"
        assert v.dtype == torch.float32, f"{k} dtype is {v.dtype}"
        assert v.ndim == 4, f"{k} has shape {tuple(v.shape)}"
        assert v.shape[0] == seq_len, f"{k} T={v.shape[0]} != {seq_len}"

    # Channel counts.
    assert item["color_lr"].shape[1]   == 3
    assert item["gt_hr"].shape[1]      == 3
    assert item["motion_lr"].shape[1]  == 2
    assert item["depth_lr"].shape[1]   == 1
    assert item["normals_lr"].shape[1] == 3
    assert item["albedo_lr"].shape[1]  == 3

    # HR/LR resolutions.
    assert tuple(item["gt_hr"].shape[2:])      == (hr_h, hr_w)
    assert tuple(item["color_lr"].shape[2:])   == (lr_h, lr_w)
    assert tuple(item["motion_lr"].shape[2:])  == (lr_h, lr_w)
    assert tuple(item["depth_lr"].shape[2:])   == (lr_h, lr_w)
    assert tuple(item["normals_lr"].shape[2:]) == (lr_h, lr_w)
    assert tuple(item["albedo_lr"].shape[2:])  == (lr_h, lr_w)


def test_noisebase_dataset_window_padding(synthetic_root: Path) -> None:
    """Sequence shorter than ``sequence_length`` should edge-pad, not error."""
    ds = NoiseBaseDataset(
        root=synthetic_root,
        sequence_length=10,   # fixture is only 4 frames
        resolution=(32, 32),
        scale_factor=2.0,
        split="train",
    )
    item = ds[0]
    assert item["gt_hr"].shape[0] == 10


def test_motion_vectors_zero_at_sequence_start(synthetic_root: Path) -> None:
    """Motion at frame 0 should be zero (no prior frame to compare against)."""
    ds = NoiseBaseDataset(
        root=synthetic_root,
        sequence_length=4,
        resolution=(32, 32),
        scale_factor=2.0,
        split="train",
    )
    item = ds[0]
    motion_t0 = item["motion_lr"][0]  # [2, H, W]
    # Frame 0 motion should be all zeros since there's no prior frame.
    assert torch.allclose(motion_t0, torch.zeros_like(motion_t0), atol=1e-5)


def test_noisebase_dataset_no_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NoiseBaseDataset(root=tmp_path, sequence_length=2)


# ---------------------------------------------------------------------------
# Real-data smoke test (skipped unless ORS_NOISEBASE_ROOT is set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ORS_NOISEBASE_ROOT"),
    reason="real NoiseBase not available; set ORS_NOISEBASE_ROOT to enable",
)
def test_noisebase_real_data() -> None:
    root = Path(os.environ["ORS_NOISEBASE_ROOT"])
    ds = NoiseBaseDataset(
        root=root,
        sequence_length=4,
        resolution=(800, 1280),
        scale_factor=2.0,
        split="train",
    )
    assert len(ds) > 0
    item = ds[0]
    assert item["gt_hr"].dtype == torch.float32
    assert item["color_lr"].shape[0] == 4

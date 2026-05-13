"""Tests for the --max-hr-crop random-crop sampler in sr_train_v7.py.

The crop is the trainer's escape hatch for fitting higher-source-HR
data (1080p, 4K Cyberpunk captures) into a consumer-GPU VRAM budget.
It must keep LR/HR registration intact and handle a small set of
edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sr_train_v7 import extract_sample_with_crop, _LR_KEYS, _HR_KEYS


def _fake_batch(B: int = 2, h_lr: int = 64, w_lr: int = 96, scale: int = 2) -> dict:
    """Synthesize a collated triplet batch with the same shape contract
    as collate_triplets()'s output, filled with a distinctive grid pattern
    so crop alignment is testable by value."""
    h_hr, w_hr = h_lr * scale, w_lr * scale
    out = {}
    # LR-shaped tensors get a grid where pixel (y, x) = (y * 1000 + x) so we
    # can verify which window was extracted.
    for k in _LR_KEYS:
        c = {"n_lr": 3, "np1_lr": 3, "n_depth": 1, "np1_depth": 1,
             "n_motion": 2, "np1_motion": 2,
             "n_normals": 3, "np1_normals": 3,
             "motion_n_to_np1": 2}[k]
        t = torch.zeros((B, c, h_lr, w_lr))
        y, x = torch.meshgrid(torch.arange(h_lr), torch.arange(w_lr), indexing="ij")
        t[..., :, :] = (y * 1000 + x).float()
        out[k] = t
    for k in _HR_KEYS:
        t = torch.zeros((B, 3, h_hr, w_hr))
        y, x = torch.meshgrid(torch.arange(h_hr), torch.arange(w_hr), indexing="ij")
        t[..., :, :] = (y * 1000 + x).float()
        out[k] = t
    return out


def test_no_crop_returns_full_sample():
    batch = _fake_batch()
    sample = extract_sample_with_crop(batch, b=0, hr_crop=None, scale=2)
    assert sample["n_lr"].shape == (1, 3, 64, 96)
    assert sample["n_gt"].shape == (1, 3, 128, 192)
    # Values unchanged
    assert torch.equal(sample["n_lr"][0], batch["n_lr"][0])


def test_crop_larger_than_hr_returns_full_sample():
    """Asking for a 999x999 crop on a 128x192 HR -> just return the full frame."""
    batch = _fake_batch()
    sample = extract_sample_with_crop(batch, b=0, hr_crop=999, scale=2)
    assert sample["n_gt"].shape == (1, 3, 128, 192)


def test_crop_returns_aligned_lr_and_hr_windows():
    """The point of the helper: HR top-left = LR top-left * scale, so LR/HR
    stay registered. Verified by reading the grid values out of the cropped
    tensors."""
    batch = _fake_batch(h_lr=64, w_lr=96, scale=2)
    g = torch.Generator().manual_seed(42)
    sample = extract_sample_with_crop(batch, b=0, hr_crop=64, scale=2, generator=g)

    # HR crop should be 64x64
    assert sample["n_gt"].shape == (1, 3, 64, 64)
    # LR crop should be 32x32
    assert sample["n_lr"].shape == (1, 3, 32, 32)
    # Recover the picked top-left from the LR crop's grid encoding
    lr_top_left = int(sample["n_lr"][0, 0, 0, 0].item())
    lr_top = lr_top_left // 1000
    lr_left = lr_top_left % 1000
    # Same for HR
    hr_top_left = int(sample["n_gt"][0, 0, 0, 0].item())
    hr_top = hr_top_left // 1000
    hr_left = hr_top_left % 1000
    # Alignment: HR top == LR top * scale
    assert hr_top == lr_top * 2
    assert hr_left == lr_left * 2


def test_crop_indivisible_by_scale_raises():
    batch = _fake_batch()
    with pytest.raises(ValueError, match="divisible by scale"):
        extract_sample_with_crop(batch, b=0, hr_crop=33, scale=2)


def test_crop_keeps_lr_only_tensors_at_lr_resolution():
    """All depth/motion/normals/motion_n_to_np1 tensors must remain at LR
    crop resolution, not somehow get HR-cropped."""
    batch = _fake_batch(h_lr=64, w_lr=96, scale=2)
    sample = extract_sample_with_crop(batch, b=1, hr_crop=64, scale=2)
    for k in _LR_KEYS:
        assert sample[k].shape[-2:] == (32, 32), (
            f"LR-key {k} should be cropped to LR-resolution 32x32; got {sample[k].shape}"
        )
    for k in _HR_KEYS:
        assert sample[k].shape[-2:] == (64, 64), (
            f"HR-key {k} should be cropped to HR-resolution 64x64; got {sample[k].shape}"
        )


def test_crop_randomness_visits_different_windows_under_different_seeds():
    """Two extracts with different seeds should land different windows
    (probabilistically). With 16 possible top positions and 32 possible
    left positions, collision odds for 5 trials are negligible."""
    batch = _fake_batch(h_lr=64, w_lr=96, scale=2)
    seen = set()
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        s = extract_sample_with_crop(batch, b=0, hr_crop=32, scale=2, generator=g)
        seen.add(int(s["n_lr"][0, 0, 0, 0].item()))
    assert len(seen) > 1, "Different seeds should visit different crop windows"


def test_crop_lr_and_hr_pixel_values_self_consistent():
    """The grid encoding lets us verify the LR and HR windows refer to the
    SAME source rectangle: the HR top-left pixel value should equal
    (LR top-left value) * (1000 * scale + scale) i.e. (lr_top*1000 + lr_left)
    scaled up to HR coords gives (lr_top*scale*1000 + lr_left*scale).
    """
    batch = _fake_batch(h_lr=64, w_lr=96, scale=2)
    g = torch.Generator().manual_seed(7)
    sample = extract_sample_with_crop(batch, b=0, hr_crop=32, scale=2, generator=g)
    lr_val = int(sample["n_lr"][0, 0, 0, 0].item())
    hr_val = int(sample["n_gt"][0, 0, 0, 0].item())
    lr_top, lr_left = lr_val // 1000, lr_val % 1000
    expected_hr = lr_top * 2 * 1000 + lr_left * 2
    assert hr_val == expected_hr

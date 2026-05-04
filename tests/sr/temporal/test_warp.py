"""Backward-warp tests for the v5 pixel temporal track."""
from __future__ import annotations

import torch

from oss.sr.temporal import upsample_motion_to_hr, warp_prev_hr


def test_upsample_motion_scales_displacement() -> None:
    motion_lr = torch.ones(1, 2, 4, 4)  # 1 LR-pixel of flow everywhere
    motion_hr = upsample_motion_to_hr(motion_lr, scale=2)
    assert motion_hr.shape == (1, 2, 8, 8)
    # LR-pixel displacement of 1 == HR-pixel displacement of 2
    assert torch.allclose(motion_hr, torch.full_like(motion_hr, 2.0), atol=1e-5)


def test_zero_motion_is_identity() -> None:
    prev_hr = torch.rand(1, 3, 16, 16)
    motion_lr = torch.zeros(1, 2, 8, 8)
    warped = warp_prev_hr(prev_hr, motion_lr, scale=2)
    assert torch.allclose(warped, prev_hr, atol=1e-5)


def test_translation_warp() -> None:
    # Convention: motion is forward flow t-1 → t.
    # Construct prev_hr with a vertical stripe at columns 4..7.
    # Forward flow x = +4 HR px means content at prev col c moved to current col c+4.
    # Backward warp at current pixel p samples prev at p − flow(p) = p − 4.
    # So at current p=8..11 we sample prev[4..7] (the stripe) → output stripe at 8..11.
    prev_hr = torch.zeros(1, 3, 16, 16)
    prev_hr[..., 4:8] = 1.0
    motion_lr = torch.zeros(1, 2, 8, 8)
    motion_lr[:, 0] = 2.0  # +2 LR px ≡ +4 HR px
    warped = warp_prev_hr(prev_hr, motion_lr, scale=2)
    assert warped[..., 8:12].mean() > 0.95
    assert warped[..., :4].mean() < 0.05

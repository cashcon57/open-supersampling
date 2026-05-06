"""Unit tests for ``temporal_consistency_loss``.

Three properties we verify:
1. Identical frames + zero motion -> loss ~= 0 (warp is identity).
2. Different frames + zero motion -> loss > 0 (no false-zero from grid_sample).
3. Gradients flow back to ``pred_t`` (so the loss can train the model).
"""
from __future__ import annotations

import torch

from oss.train.losses import temporal_consistency_loss


def test_temporal_consistency_zero_motion_identical_frames() -> None:
    pred = torch.rand(1, 3, 32, 32)
    motion = torch.zeros(1, 2, 16, 16)
    loss = temporal_consistency_loss(pred, pred, motion, scale_factor=2.0)
    assert loss.item() < 1e-3


def test_temporal_consistency_zero_motion_different_frames() -> None:
    pred_t = torch.rand(1, 3, 32, 32)
    pred_prev = torch.rand(1, 3, 32, 32)
    motion = torch.zeros(1, 2, 16, 16)
    loss = temporal_consistency_loss(pred_t, pred_prev, motion, scale_factor=2.0)
    assert loss.item() > 0


def test_temporal_consistency_grad_flow() -> None:
    pred_t = torch.rand(1, 3, 32, 32, requires_grad=True)
    pred_prev = torch.rand(1, 3, 32, 32)
    motion = torch.randn(1, 2, 16, 16) * 0.1
    loss = temporal_consistency_loss(pred_t, pred_prev, motion, scale_factor=2.0)
    loss.backward()
    assert pred_t.grad is not None


def test_warp_uses_forward_flow_convention() -> None:
    """Audit-fixed convention (2026-05-06): motion is FORWARD flow t-1 → t.
    A constant +N-pixel x-motion means "every pixel in frame t-1 moved N
    pixels right in frame t." Warping image_{t-1} into frame-t coords
    should pull the source content LEFT by N pixels (so the right edge
    is filled by border padding and the original content shifts left).
    """
    from oss.train.losses import warp_with_motion

    # Single bright pixel in image_prev at (y=8, x=4). HR is 16x16.
    image_prev = torch.zeros(1, 1, 16, 16)
    image_prev[0, 0, 8, 4] = 1.0

    # Forward motion: +4 pixels in x at HR scale -> +2 in LR (scale=2).
    # All pixels in t-1 ENDED UP +4 right in t. So warp(image_prev) should
    # have the bright pixel at x=4+4=8 in frame-t coords... wait, no.
    # warp(image_prev, fwd) returns image_t-aligned: for each pixel x_t,
    # output[x_t] = image_prev[x_t - fwd(x_t)]. Pixel x_t=8 reads
    # image_prev[8 - 4 = 4] which is our bright pixel. So bright pixel
    # APPEARS at x_t = 8 in the warped output.
    motion = torch.zeros(1, 2, 8, 8)
    motion[0, 0, :, :] = 2.0  # LR-pixel x-displacement -> 4 HR-pixels
    warped = warp_with_motion(image_prev, motion, scale_factor=2.0)

    # Bright pixel should appear at (8, 8) in warped output, not (8, 0)
    # (which would be the OLD wrong-direction sample).
    peak = warped[0, 0]
    max_val, max_idx = peak.flatten().max(0)
    py, px = int(max_idx // 16), int(max_idx % 16)
    assert max_val.item() > 0.5
    assert (py, px) == (8, 8), (
        f"Forward-flow warp should put bright pixel at (8,8); got ({py},{px}). "
        f"If this fails, the sign in warp_with_motion sample_grid is inverted."
    )

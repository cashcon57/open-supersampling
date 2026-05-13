"""Tests for the v7 OSS-FX loss recipe."""
from __future__ import annotations

import pytest
import torch

from oss.sr.v7.losses import (
    charbonnier,
    warp_image_by_motion,
    oss_fx_loss,
)


def test_charbonnier_zero_when_pred_equals_target():
    pred = torch.rand((2, 3, 8, 8))
    loss = charbonnier(pred, pred)
    # Charbonnier is sqrt(0 + eps^2) = eps when prediction matches target.
    assert loss.item() < 2e-6


def test_charbonnier_decreases_as_prediction_approaches_target():
    pred = torch.zeros((1, 3, 4, 4))
    target = torch.ones((1, 3, 4, 4))
    far = charbonnier(pred, target).item()
    close = charbonnier(0.5 * target, target).item()
    closer = charbonnier(0.9 * target, target).item()
    assert far > close > closer


def test_warp_image_by_motion_identity_motion_returns_image():
    image = torch.rand((1, 3, 16, 16))
    motion_lr = torch.zeros((1, 2, 8, 8))
    warped = warp_image_by_motion(image, motion_lr, scale=2)
    # With zero motion, warping should be near-identity (small diff
    # from grid_sample interpolation at boundaries).
    diff = (warped - image).abs().mean().item()
    assert diff < 0.01


def test_warp_image_by_motion_shifts_with_uniform_motion():
    """Uniform motion field of (1, 0) px shifts the image one pixel
    to the right."""
    image = torch.zeros((1, 3, 8, 8))
    image[0, :, 4, 3] = 1.0   # bright pixel at (4, 3)
    # Motion at LR is 0.5 px (HR scale=2 -> 1 HR px). Direction: +x.
    motion_lr = torch.zeros((1, 2, 4, 4))
    motion_lr[0, 0] = 0.5
    warped = warp_image_by_motion(image, motion_lr, scale=2)
    # Bright pixel should now be at (4, 4) since we're sampling at
    # (pixel - motion). At (4, 4) we sample image at (4-1, 4) = (4, 3)
    # which had the bright value.
    assert warped[0, 0, 4, 4].item() > 0.8


def test_oss_fx_loss_returns_components_and_total():
    out_main = torch.rand((1, 3, 8, 8))
    gt_main = torch.rand((1, 3, 8, 8))
    loss, parts = oss_fx_loss(
        out_main, gt_main,
        out_inter_list=None, gt_inter_list=None,
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,   # disable LPIPS for this test so we don't depend on the package
        lambda_fg=0.0,
        lambda_temp_consistency=0.0,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert "sr_charbonnier" in parts
    assert "total" in parts
    assert "fg_charbonnier" not in parts   # FG disabled


def test_oss_fx_loss_includes_fg_when_intermediates_given():
    out_main = torch.rand((1, 3, 8, 8))
    gt_main = torch.rand((1, 3, 8, 8))
    out_inter = torch.rand((1, 3, 8, 8))
    gt_inter = torch.rand((1, 3, 8, 8))
    loss, parts = oss_fx_loss(
        out_main, gt_main,
        out_inter_list=[out_inter], gt_inter_list=[gt_inter],
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,
        lambda_fg=1.0,
        lambda_fg_lpips=0.0,
        lambda_temp_consistency=0.0,
    )
    assert "fg_charbonnier" in parts
    # Total should include both SR + FG contributions
    assert parts["total"] == pytest.approx(parts["sr_charbonnier"] + parts["fg_charbonnier"], rel=1e-5)


def test_oss_fx_loss_includes_temporal_consistency_when_prev_given():
    out_main = torch.rand((1, 3, 8, 8))
    gt_main = torch.rand((1, 3, 8, 8))
    out_prev = torch.rand((1, 3, 8, 8))
    motion = torch.zeros((1, 2, 4, 4))
    loss, parts = oss_fx_loss(
        out_main, gt_main,
        out_prev_for_consistency=out_prev,
        motion_lr_prev_to_curr=motion,
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,
        lambda_temp_consistency=1.0,
    )
    assert "temp_consistency" in parts


def test_oss_fx_loss_gradients_flow():
    out_main = torch.rand((1, 3, 8, 8), requires_grad=True)
    gt_main = torch.rand((1, 3, 8, 8))
    loss, _ = oss_fx_loss(
        out_main, gt_main,
        lambda_charbonnier=1.0,
        lambda_lpips=0.0,
        lambda_fg=0.0,
        lambda_temp_consistency=0.0,
    )
    loss.backward()
    assert out_main.grad is not None
    assert out_main.grad.abs().sum().item() > 0.0

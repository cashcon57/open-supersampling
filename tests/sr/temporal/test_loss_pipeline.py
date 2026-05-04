"""End-to-end loss pipeline integration test for v5-pixel-temporal."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oss.sr.temporal import TemporalSRModel
from oss.train.losses import temporal_consistency_loss


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Tiny box-window SSIM stand-in matching oss.train.losses style."""
    mu_p = F.avg_pool2d(pred, 3, 1, 1)
    mu_t = F.avg_pool2d(target, 3, 1, 1)
    var_p = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_p * mu_p
    var_t = F.avg_pool2d(target * target, 3, 1, 1) - mu_t * mu_t
    cov = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
    )
    return 1.0 - ssim.clamp(0, 1).mean()


def test_full_loss_pipeline_grads() -> None:
    torch.manual_seed(0)
    model = TemporalSRModel(in_channels=12, scale=2, tier="standard")

    # Synthetic batch — two consecutive frames.
    lr = 8
    inputs_t = {
        "lr_inputs": torch.rand(1, 12, lr, lr),
        "prev_hr": torch.rand(1, 3, lr * 2, lr * 2),
        "depth_hr_curr": torch.rand(1, 1, lr * 2, lr * 2),
        "depth_hr_prev": torch.rand(1, 1, lr * 2, lr * 2),
        "motion_lr": torch.randn(1, 2, lr, lr) * 0.1,
    }
    inputs_tp1 = {
        "lr_inputs": torch.rand(1, 12, lr, lr),
        "prev_hr": None,  # filled below from out_t
        "depth_hr_curr": torch.rand(1, 1, lr * 2, lr * 2),
        "depth_hr_prev": inputs_t["depth_hr_curr"],
        "motion_lr": torch.randn(1, 2, lr, lr) * 0.1,
    }
    gt_hr_t = torch.rand(1, 3, lr * 2, lr * 2)
    gt_hr_tp1 = torch.rand(1, 3, lr * 2, lr * 2)
    motion_t_to_tp1 = torch.randn(1, 2, lr, lr) * 0.1

    out_t = model(**inputs_t)
    inputs_tp1["prev_hr"] = out_t.detach()
    out_tp1 = model(**inputs_tp1)

    w_l1, w_ssim, w_lpips, w_tc = 1.0, 0.1, 0.1, 0.05
    appearance = (
        w_l1 * F.l1_loss(out_t, gt_hr_t)
        + w_ssim * _ssim_loss(out_t, gt_hr_t)
        + w_l1 * F.l1_loss(out_tp1, gt_hr_tp1)
        + w_ssim * _ssim_loss(out_tp1, gt_hr_tp1)
    )
    # LPIPS gated on package presence so test still runs in minimal envs.
    try:
        import lpips  # type: ignore[import-not-found]
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        def _lp(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return lpips_fn(p * 2.0 - 1.0, t * 2.0 - 1.0).mean()
        appearance = appearance + w_lpips * (_lp(out_t, gt_hr_t) + _lp(out_tp1, gt_hr_tp1))
    except Exception:
        pass  # lpips not installed; loss runs without it
    tc = temporal_consistency_loss(out_tp1, out_t, motion_t_to_tp1, scale_factor=2.0)
    loss = appearance + w_tc * tc

    assert torch.isfinite(loss)
    loss.backward()
    for group_name, params in (
        ("head", model.head.parameters()),
        ("gate", model.gate.parameters()),
        ("backbone", model.backbone.parameters()),
    ):
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all(), group_name

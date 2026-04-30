"""Composite training loss: relative-L2 + (1 - SSIM) + LPIPS.

Rationale:
- ``relative_l2`` (Vogels 2018 / Intel OIDN-style) prevents bright pixels from
  dominating the gradient — essential for HDR path-traced training data.
- SSIM keeps local structure crisp; we use a simplified box-window estimate
  that is sufficient for training-time supervision (Wang 2004 11x11 Gaussian
  is overkill here and adds dependencies).
- LPIPS (Zhang 2018) optionally adds a perceptual term. Lazy-imported and
  parameter-frozen; skipped entirely when ``w_lpips=0``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Vogels 2018 relative-L2: ``((p - t)^2 / (t.detach()^2 + eps)).mean()``.

    The detach prevents the denominator from contributing to gradients (would
    otherwise pull predictions toward zero in dark regions).
    """
    denom = target.detach().pow(2) + eps
    return ((pred - target).pow(2) / denom).mean()


def _ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Simplified per-channel SSIM with a box window (avg_pool2d).

    Returns a scalar mean-SSIM in roughly [0, 1]. Sufficient for training
    supervision; not a benchmark-grade implementation.
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window // 2

    mu_p = F.avg_pool2d(pred,   window, stride=1, padding=pad)
    mu_t = F.avg_pool2d(target, window, stride=1, padding=pad)

    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2 = F.avg_pool2d(pred * pred,     window, stride=1, padding=pad) - mu_p2
    sigma_t2 = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu_t2
    sigma_pt = F.avg_pool2d(pred * target,   window, stride=1, padding=pad) - mu_pt

    num = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    den = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    return (num / den).mean()


def _tonemap(x: torch.Tensor) -> torch.Tensor:
    """Reinhard-style x/(x+1). Bounds HDR into [0,1] for SSIM/LPIPS sanity."""
    return x / (x + 1.0)


class CompositeLoss(nn.Module):
    """``w_l2 * relative_l2 + w_ssim * (1 - SSIM) + w_lpips * LPIPS``.

    LPIPS is lazy-imported only when ``w_lpips > 0`` so smoke tests on CPU
    don't pay the VGG download/load cost.
    """

    def __init__(
        self,
        w_l2: float = 1.0,
        w_ssim: float = 0.1,
        w_lpips: float = 0.05,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.w_l2 = float(w_l2)
        self.w_ssim = float(w_ssim)
        self.w_lpips = float(w_lpips)
        self._lpips_net_name = lpips_net
        self._lpips: Optional[nn.Module] = None
        if self.w_lpips > 0:
            self._init_lpips()

    def _init_lpips(self) -> None:
        import lpips as _lpips_pkg  # local import keeps smoke tests dependency-free
        net = _lpips_pkg.LPIPS(net=self._lpips_net_name, verbose=False)
        for p in net.parameters():
            p.requires_grad = False
        net.train(False)
        self._lpips = net

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_l2 * relative_l2(pred, target)

        if self.w_ssim > 0 or self.w_lpips > 0:
            pred_ldr   = _tonemap(pred.clamp(min=0.0))
            target_ldr = _tonemap(target.clamp(min=0.0))

            if self.w_ssim > 0:
                loss = loss + self.w_ssim * (1.0 - _ssim(pred_ldr, target_ldr))

            if self.w_lpips > 0:
                assert self._lpips is not None
                # LPIPS expects [-1, 1]
                p = pred_ldr * 2.0 - 1.0
                t = target_ldr * 2.0 - 1.0
                loss = loss + self.w_lpips * self._lpips(p, t).mean()

        return loss

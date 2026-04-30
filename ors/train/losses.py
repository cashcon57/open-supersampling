"""Composite training loss: relative-L2 + (1 - SSIM) + LPIPS + wavelet.

Rationale:
- ``relative_l2`` (Vogels 2018 / Intel OIDN-style) prevents bright pixels from
  dominating the gradient — essential for HDR path-traced training data.
- SSIM keeps local structure crisp; we use a simplified box-window estimate
  that is sufficient for training-time supervision (Wang 2004 11x11 Gaussian
  is overkill here and adds dependencies).
- LPIPS (Zhang 2018) optionally adds a perceptual term. Lazy-imported and
  parameter-frozen; skipped entirely when ``w_lpips=0``.
- ``wavelet_loss`` (Poudel 2025) supervises high-frequency subbands explicitly.
  Pixel-space L2 underweights edges and texture — by computing L1 in SWT space
  and weighting detail subbands more heavily than the LL approximation, we
  get a direct gradient signal on the subbands the wavelet head predicts.
  Optional and skipped when ``w_wavelet=0``.
"""
from __future__ import annotations

from typing import Optional, Sequence

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


def temporal_consistency_loss(
    pred_t: torch.Tensor,
    pred_prev: torch.Tensor,
    motion_lr: torch.Tensor,
    scale_factor: float = 2.0,
) -> torch.Tensor:
    """Penalize per-pixel difference between current frame's prediction and
    the motion-warped previous prediction. Encourages temporal stability.

    Args:
        pred_t:       (B, 3, H_hr, W_hr) — current frame's HR prediction.
        pred_prev:    (B, 3, H_hr, W_hr) — previous frame's HR prediction.
        motion_lr:    (B, 2, H_lr, W_lr) — LR motion vectors (channel 0=x, 1=y),
                      pointing from current frame to previous frame in pixel
                      displacements at LR scale.
        scale_factor: HR / LR ratio (default 2.0).

    Returns:
        Scalar L1 of (pred_t - warp(pred_prev, motion)).

    Implementation notes:
        - Motion vectors are bilinearly upsampled to HR and scaled by
          ``scale_factor`` to convert LR-pixel displacements to HR-pixel
          displacements.
        - The HR displacements are then normalized into ``grid_sample``'s
          [-1, 1] coordinate system by ``2 / extent``.
        - We clamp the sample grid to a broad [-2, 2] range to avoid NaN
          from extreme motion; ``padding_mode='border'`` handles out-of-bounds.
    """
    B, _, H_hr, W_hr = pred_t.shape

    motion_hr = F.interpolate(
        motion_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False
    )
    motion_hr = motion_hr * scale_factor  # LR-pixel disp -> HR-pixel disp

    # Identity sampling grid in normalized [-1, 1] coords. With
    # ``align_corners=False`` (which we use below to match the rest of the
    # pipeline's interpolate calls), pixel centers lie at
    # ``(2i + 1)/N - 1`` rather than ``linspace(-1, 1, N)``. Using the wrong
    # convention here introduces a half-pixel shift that breaks the
    # zero-motion-identical-frames invariant.
    iy = (torch.arange(H_hr, device=pred_t.device, dtype=pred_t.dtype) + 0.5) * (
        2.0 / H_hr
    ) - 1.0
    ix = (torch.arange(W_hr, device=pred_t.device, dtype=pred_t.dtype) + 0.5) * (
        2.0 / W_hr
    ) - 1.0
    yy, xx = torch.meshgrid(iy, ix, indexing="ij")
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # (B, 2, H, W) -> (B, H, W, 2). Channel 0 = x-disp, channel 1 = y-disp.
    motion_grid = motion_hr.permute(0, 2, 3, 1)
    # Normalize HR-pixel displacements to grid-coord displacements.
    extent = torch.tensor(
        [W_hr, H_hr], device=pred_t.device, dtype=pred_t.dtype
    )
    motion_grid_norm = motion_grid * 2.0 / extent

    sample_grid = (base_grid + motion_grid_norm).clamp(-2, 2)

    pred_prev_warped = F.grid_sample(
        pred_prev,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return (pred_t - pred_prev_warped).abs().mean()


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


def wavelet_loss(
    pred_rgb_hr: torch.Tensor,
    target_rgb_hr: torch.Tensor,
    levels: int = 2,
    wavelet: str = "db2",
    weights: Sequence[float] = (1.0, 2.0, 2.0),
) -> torch.Tensor:
    """Subband-weighted L1 in SWT space (Poudel 2025).

    Decomposes both ``pred`` and ``target`` with a stationary wavelet
    transform at ``levels`` and accumulates an L1 distance per subband. The
    LL (approximation) coefficient is weighted by ``weights[0]``; the level-
    ``j`` detail subbands (LH / HL / HH) by ``weights[j]``. Default
    ``(1, 2, 2)`` follows the paper: high-frequency subbands carry the
    edge / texture detail that pixel-space L2 underweights, so we double
    their loss contribution.

    Args:
        pred_rgb_hr: ``(B, 3, H, W)`` predicted HR RGB.
        target_rgb_hr: ``(B, 3, H, W)`` ground-truth HR RGB.
        levels: SWT decomposition depth (2 matches ORU-Pico's wavelet head).
        wavelet: Wavelet family (default ``'db2'``).
        weights: ``(w_LL, w_lvl1, w_lvl2, ...)`` length ``levels + 1``.

    Returns:
        Scalar L1 loss.
    """
    # Lazy-import the SWT primitive so this loss module stays decoupled from
    # the model package's import-time pytorch-wavelets dependency check.
    from ors.model.wavelet import SWT2D

    if len(weights) != levels + 1:
        raise ValueError(
            f"weights must have length levels+1={levels + 1}, got {len(weights)}"
        )

    # Build the SWT once per call. We don't cache it — the buffer cost is
    # tiny (the four 4-tap filter values) and caching would tie the loss to
    # a specific device / dtype.
    swt = SWT2D(levels=levels, wavelet=wavelet).to(
        device=pred_rgb_hr.device, dtype=pred_rgb_hr.dtype
    )

    # Run on detached target — gradient should only flow through the
    # prediction (target is the supervision signal).
    pred_ll, pred_details = swt(pred_rgb_hr)
    with torch.no_grad():
        target_ll, target_details = swt(target_rgb_hr)

    loss = weights[0] * (pred_ll - target_ll).abs().mean()
    for j in range(levels):
        w_j = weights[j + 1]
        for p, t in zip(pred_details[j], target_details[j]):
            loss = loss + w_j * (p - t).abs().mean()
    return loss


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
        w_wavelet: float = 0.0,
        wavelet_levels: int = 2,
        wavelet_name: str = "db2",
        wavelet_weights: Sequence[float] = (1.0, 2.0, 2.0),
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.w_l2 = float(w_l2)
        self.w_ssim = float(w_ssim)
        self.w_lpips = float(w_lpips)
        self.w_wavelet = float(w_wavelet)
        self.wavelet_levels = wavelet_levels
        self.wavelet_name = wavelet_name
        self.wavelet_weights = tuple(wavelet_weights)
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

        if self.w_wavelet > 0:
            # Wavelet loss runs on the raw HDR tensors — the SWT is linear
            # so tonemapping isn't required for sane gradients, and we want
            # the head supervised in the same space it predicts in.
            loss = loss + self.w_wavelet * wavelet_loss(
                pred,
                target,
                levels=self.wavelet_levels,
                wavelet=self.wavelet_name,
                weights=self.wavelet_weights,
            )

        return loss

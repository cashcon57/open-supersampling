"""Extrapolation losses for OSS-FX α-conditioned frame prediction."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _ssim(pred: Tensor, target: Tensor, window: int = 11) -> Tensor:
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


def extrapolation_loss(
    pred: Tensor,
    target: Tensor,
    pred_prev: Tensor,
    alpha: Tensor,
    w_l1: float = 1.0,
    w_ssim: float = 0.1,
    w_temporal: float = 0.05,
) -> Tensor:
    l1 = (pred - target).abs().mean()

    pred_ldr = pred.clamp(0.0, 1.0)
    target_ldr = target.clamp(0.0, 1.0)
    ssim_val = _ssim(pred_ldr, target_ldr)
    ssim_term = 1.0 - ssim_val

    alpha_mean = alpha.mean().clamp(min=1e-6)
    temporal_weight = alpha_mean.pow(2)
    temporal_diff = ((pred - pred_prev) / alpha_mean).abs().mean()
    temporal_term = temporal_diff * (1.0 - temporal_weight)

    return w_l1 * l1 + w_ssim * ssim_term + w_temporal * temporal_term

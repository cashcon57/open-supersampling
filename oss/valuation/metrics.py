"""Image-quality metrics for ORS valuation.

PSNR computed in linear (HDR) space; SSIM/LPIPS in tonemapped LDR space (caller
tonemaps before invoking these).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LPIPS_MODEL = None


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return torch.tensor(99.0)
    return -10 * torch.log10(mse)


def _ssim_window():
    return torch.ones(1, 1, 11, 11) / 121.0


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simplified SSIM, single 11x11 box window (sufficient for MVP)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = _ssim_window().to(pred.device).repeat(pred.shape[1], 1, 1, 1)
    kw = dict(stride=1, padding=5, groups=pred.shape[1])
    mu_x = F.conv2d(pred, win, **kw)
    mu_y = F.conv2d(target, win, **kw)
    sigma_x = F.conv2d(pred * pred, win, **kw) - mu_x ** 2
    sigma_y = F.conv2d(target * target, win, **kw) - mu_y ** 2
    sigma_xy = F.conv2d(pred * target, win, **kw) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (ssim_n / ssim_d.clamp(min=1e-8)).mean()


def lpips_dist(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """LPIPS distance (VGG backbone). Inputs expected in [-1, 1]."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips as _lpips
        _LPIPS_MODEL = _lpips.LPIPS(net="vgg")
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad = False
    _LPIPS_MODEL = _LPIPS_MODEL.to(pred.device)
    return _LPIPS_MODEL(pred, target).mean()

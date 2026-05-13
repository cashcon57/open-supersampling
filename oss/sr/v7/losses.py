"""v7 OSS-FX loss recipe.

Composes:
  L_sr     = Charbonnier(out_at_t=N, GT_N) + lambda_lpips * LPIPS_VGG(...)
  L_fg     = Charbonnier(out_at_t=N+alpha, GT_at_t=N+alpha) + perceptual
  L_temp   = mean | out_at_t=N - warp(out_at_t=N-1, motion) |   (training-time
              temporal-consistency proxy; no GT needed for this)

The relative weights are configurable; the FG branch can be disabled by
setting lambda_fg = 0 (e.g. for the first ~20K warmup steps where the
model learns SR at t=N only, then alpha<1 supervision turns on).

Designed to be used from a training loop where each step produces:
  out_main        (B, 3, H, W)  prediction at t = current frame
  out_inter_list  list of (B, 3, H, W) predictions at t = N + alpha for
                  alpha in args.intermediate_alphas
  gt_main         (B, 3, H, W)
  gt_inter_list   list of (B, 3, H, W) ground-truth intermediate frames

Returns a (scalar_loss, dict_of_components) tuple for logging.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Charbonnier loss, optionally weighted per-pixel.

    weight: (B, 1, H, W) or broadcastable — multiplied into the per-pixel
        residual before averaging. Used by RRM (Random Reshading Masking)
        to put 2x emphasis on synthetic disocclusion regions.
    """
    residual = torch.sqrt((pred - target) ** 2 + eps * eps)
    if weight is not None:
        residual = residual * weight
    return residual.mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """3x3 Sobel kernels for x and y gradients, shaped for grouped conv on
    a 3-channel image (one kernel per channel)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)
    # (out_ch=3, in_ch_per_group=1, 3, 3) for groups=3 grouped conv
    kx = kx.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    ky = ky.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    return kx, ky


def sobel_grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """High-frequency edge loss: L1 on Sobel gradient magnitudes.

    Penalises differences in *edge structure* between pred and target,
    not just per-pixel brightness. Helps preserve thin geometry + crisp
    boundaries at the cost of being more sensitive to noise. Use a
    small weight (e.g. 0.1) so it doesn't dominate the Charbonnier
    pixel loss.

    Both pred and target are (B, 3, H, W) in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    kx, ky = _sobel_kernels(pred.device, pred.dtype)
    # Grouped conv: per-channel x and y gradients
    gx_p = F.conv2d(pred, kx, padding=1, groups=3)
    gy_p = F.conv2d(pred, ky, padding=1, groups=3)
    gx_t = F.conv2d(target, kx, padding=1, groups=3)
    gy_t = F.conv2d(target, ky, padding=1, groups=3)
    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-8)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-8)
    return (mag_p - mag_t).abs().mean()


def _to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0


class _LazyLPIPS:
    """Wraps the LPIPS-VGG net lazily so importing this module doesn't
    trigger a 100 MB download / hub fetch at import time."""
    _instance = None

    @classmethod
    def get(cls, device: torch.device) -> Optional[nn.Module]:
        if cls._instance is False:
            return None
        if cls._instance is None:
            try:
                import lpips
                cls._instance = lpips.LPIPS(net="vgg", verbose=False).to(device)
                cls._instance.train(False)
                for p in cls._instance.parameters():
                    p.requires_grad_(False)
            except Exception:
                cls._instance = False
                return None
        return cls._instance


def lpips_vgg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """LPIPS-VGG perceptual loss. Inputs in [0, 1] (RGB); rescaled to
    [-1, 1] internally to match the lpips package convention. Returns
    a scalar averaged over the batch."""
    model = _LazyLPIPS.get(pred.device)
    if model is None:
        # Fallback: zero loss if lpips not available. Caller can set
        # lambda_lpips=0 if they want to skip.
        return pred.new_zeros(())
    p = _to_pm1(pred)
    t = _to_pm1(target)
    return model(p, t).mean()


def warp_image_by_motion(
    image: torch.Tensor,   # (B, 3, H, W) -- the image AT TIME t (to warp forward)
    motion_lr: torch.Tensor,   # (B, 2, H_lr, W_lr) -- motion field from t to t+1 (LR-scale)
    scale: int = 2,
) -> torch.Tensor:
    """Warp image by motion vectors. motion_lr is LR-resolution; we
    upsample to HR via bilinear and use grid_sample to sample image at
    (pixel - motion). Returns the predicted image at t+1.

    Pure utility; no learnable parameters.
    """
    b, _, h_hr, w_hr = image.shape
    motion_hr = F.interpolate(motion_lr, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
    motion_hr = motion_hr * float(scale)

    # Build pixel grid (B, H, W, 2) in normalized coords [-1, 1].
    device = image.device
    dtype = image.dtype
    ys = torch.arange(h_hr, device=device, dtype=dtype)
    xs = torch.arange(w_hr, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)   # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    # Sample at (pixel - motion) so we're picking up where each pixel
    # CAME FROM at time t.
    src_x = grid_x - motion_hr[:, 0]
    src_y = grid_y - motion_hr[:, 1]
    src_x_n = (src_x / max(w_hr - 1, 1)) * 2.0 - 1.0
    src_y_n = (src_y / max(h_hr - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([src_x_n, src_y_n], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def oss_fx_loss(
    out_main: torch.Tensor,                            # (B, 3, H, W)
    gt_main: torch.Tensor,                             # (B, 3, H, W)
    out_inter_list: Sequence[torch.Tensor] | None = None,   # list of (B, 3, H, W)
    gt_inter_list: Sequence[torch.Tensor] | None = None,    # list of (B, 3, H, W)
    out_prev_for_consistency: Optional[torch.Tensor] = None,  # (B, 3, H, W), the prev-frame output
    motion_lr_prev_to_curr: Optional[torch.Tensor] = None,    # (B, 2, H_lr, W_lr)
    lambda_charbonnier: float = 1.0,
    lambda_lpips: float = 1.0,
    lambda_fg: float = 1.0,
    lambda_fg_lpips: float = 0.5,
    lambda_temp_consistency: float = 0.1,
    lambda_sobel: float = 0.0,
    rrm_weight_main: Optional[torch.Tensor] = None,
    rrm_weight_inter: Optional[torch.Tensor] = None,
    scale_for_warp: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the v7 OSS-FX loss + a component breakdown for logging.

    Returns (total_loss, components_dict).
    """
    parts: dict[str, float] = {}
    loss_total = out_main.new_zeros(())

    # SR loss at t = N
    l_char_sr = charbonnier(out_main, gt_main, weight=rrm_weight_main)
    parts["sr_charbonnier"] = float(l_char_sr.item())
    loss_total = loss_total + lambda_charbonnier * l_char_sr

    if lambda_lpips > 0.0:
        l_lpips_sr = lpips_vgg(out_main, gt_main)
        parts["sr_lpips"] = float(l_lpips_sr.item())
        loss_total = loss_total + lambda_lpips * l_lpips_sr

    # High-frequency edge term: penalises Sobel-gradient differences.
    # Off by default (lambda_sobel=0). Recommended ~0.1 for the Heavy /
    # Standard teacher runs where preserving thin geometry matters.
    if lambda_sobel > 0.0:
        l_sobel = sobel_grad_l1(out_main, gt_main)
        parts["sr_sobel"] = float(l_sobel.item())
        loss_total = loss_total + lambda_sobel * l_sobel

    # FG loss at t = N + alpha for each intermediate
    if out_inter_list and gt_inter_list and lambda_fg > 0.0:
        if len(out_inter_list) != len(gt_inter_list):
            raise ValueError(
                f"out_inter_list ({len(out_inter_list)}) and gt_inter_list "
                f"({len(gt_inter_list)}) must have matching length"
            )
        fg_char_sum = out_main.new_zeros(())
        fg_lpips_sum = out_main.new_zeros(())
        for out_inter, gt_inter in zip(out_inter_list, gt_inter_list):
            fg_char_sum = fg_char_sum + charbonnier(
                out_inter, gt_inter, weight=rrm_weight_inter,
            )
            if lambda_fg_lpips > 0.0:
                fg_lpips_sum = fg_lpips_sum + lpips_vgg(out_inter, gt_inter)
        n = float(len(out_inter_list))
        l_fg_char = fg_char_sum / n
        parts["fg_charbonnier"] = float(l_fg_char.item())
        loss_total = loss_total + lambda_fg * l_fg_char
        if lambda_fg_lpips > 0.0:
            l_fg_lpips = fg_lpips_sum / n
            parts["fg_lpips"] = float(l_fg_lpips.item())
            loss_total = loss_total + lambda_fg * lambda_fg_lpips * l_fg_lpips

    # Temporal-consistency loss between consecutive SR frames
    if (
        out_prev_for_consistency is not None
        and motion_lr_prev_to_curr is not None
        and lambda_temp_consistency > 0.0
    ):
        warped_prev = warp_image_by_motion(
            out_prev_for_consistency,
            motion_lr_prev_to_curr,
            scale=scale_for_warp,
        )
        l_temp = (warped_prev - out_main).abs().mean()
        parts["temp_consistency"] = float(l_temp.item())
        loss_total = loss_total + lambda_temp_consistency * l_temp

    parts["total"] = float(loss_total.item())
    return loss_total, parts

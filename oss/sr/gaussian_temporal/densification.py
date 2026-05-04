"""Heuristic residual-driven densification for v5 Gaussian-temporal.

Soft top-K (Gumbel-Softmax) is post-v5 per spec §risks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from oss.sr.gaussian_temporal.gaussian_field import GaussianField


def densify(
    field: GaussianField,
    lr_target: torch.Tensor,
    rendered: torch.Tensor,
    tile_size: int,
    residual_threshold: float,
    max_new: int,
) -> GaussianField:
    """Spawn new Gaussians at high-residual tile centers.

    Color is the tile mean of ``lr_target`` (gradient flows back to lr_target);
    position/scale/rotation/opacity are detached scalar inits (no gradient).
    """
    if lr_target.shape != rendered.shape:
        raise ValueError("lr_target and rendered must have same shape")
    b, c, h, w = lr_target.shape
    if b != 1:
        # GaussianField is per-sample state; batched fields are per-sample
        # lists, not stacked tensors. Densify must run per item.
        raise ValueError(f"densify expects B=1; got {b}. Loop in caller for batches.")
    tiles_h, tiles_w = h // tile_size, w // tile_size

    residual = (lr_target - rendered).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    pooled = F.avg_pool2d(residual, kernel_size=tile_size, stride=tile_size)  # (B, 1, tH, tW)
    flat = pooled.view(-1)
    above = (flat > residual_threshold).nonzero(as_tuple=True)[0]
    if above.numel() == 0:
        return field
    if above.numel() > max_new:
        # Take the top-K by residual magnitude.
        scores = flat[above]
        topk = torch.topk(scores, k=max_new).indices
        above = above[topk]

    out = field.clone()
    free_slots = (~out.alive).nonzero(as_tuple=True)[0]
    n_to_insert = min(above.numel(), free_slots.numel())
    if n_to_insert == 0:
        return out
    target_slots = free_slots[:n_to_insert]
    chosen = above[:n_to_insert]

    tile_y = chosen // tiles_w
    tile_x = chosen % tiles_w
    cx = (tile_x.float() + 0.5) * tile_size
    cy = (tile_y.float() + 0.5) * tile_size

    # Tile mean color (gradient flows here).
    pooled_color = F.avg_pool2d(lr_target, kernel_size=tile_size, stride=tile_size)  # (B, 3, tH, tW)
    pooled_color_flat = pooled_color.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*tH*tW, 3)
    inserted_color = pooled_color_flat[chosen]

    # Insertion (detached scalar inits, except color).
    out.mu = out.mu.clone()
    out.log_scale = out.log_scale.clone()
    out.rotation = out.rotation.clone()
    out.opacity = out.opacity.clone()
    out.alive = out.alive.clone()
    out.color = out.color.clone()

    out.mu[target_slots, 0] = cx.detach()
    out.mu[target_slots, 1] = cy.detach()
    out.log_scale[target_slots] = 0.0
    out.rotation[target_slots] = 0.0
    out.opacity[target_slots] = 0.5
    out.color[target_slots] = inserted_color  # gradient flows to lr_target via this assignment
    out.alive[target_slots] = True
    return out


__all__ = ["densify"]

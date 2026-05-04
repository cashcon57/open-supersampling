"""Thin wrapper that turns a `GaussianField` into a `GaussianBatch` and renders
it through the existing OSS Gaussian rasterizer.

Why this lives here (not in `oss/gaussian/renderer/`):
- The renderer is a generic 2D Gaussian rasterizer; it has no notion of the
  v5 SR-specific `GaussianField` (alive mask, opacity-as-separate-channel,
  history). Keeping the field-specific bookkeeping in this thin wrapper keeps
  the renderer module decoupled from SR-side concerns.

Renderer API contract (confirmed against `oss/gaussian/renderer/rasterizer.py`):
- `GaussianBatch(xy, scale, rot, feat)` — no `opacities` field; opacity is
  folded into `feat` (multiplied per-Gaussian).
- `Rasterizer().__call__(gaussians, output_hw) -> (F, H, W)`.
"""
from __future__ import annotations

import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer
from oss.sr.gaussian_temporal.gaussian_field import GaussianField

# Module-level singleton: `Rasterizer.__init__` runs the (cheap-but-nontrivial)
# backend selection logic and we don't want to pay it on every render call.
_RASTERIZER = Rasterizer()


def render_field(field: GaussianField, output_hw: tuple[int, int]) -> torch.Tensor:
    """Render a `GaussianField` at `output_hw` resolution.

    Returns:
        Tensor of shape ``(1, F, H, W)``. ``F = 3`` (RGB) for v5.

    Notes:
        - Only alive Gaussians contribute (dead rows are masked out before the
          batch is built).
        - Opacity is multiplied into the per-Gaussian `feat` so the renderer's
          alpha-blend sees the right contribution (the underlying renderer has
          no separate `opacities` argument).
        - Empty alive set: returns a device/dtype-correct zero tensor without
          calling the rasterizer (which would otherwise reject an empty batch).
    """
    alive = field.alive
    n = int(alive.sum().item())
    if n == 0:
        h, w = output_hw
        return torch.zeros(1, 3, h, w, device=field.mu.device, dtype=field.mu.dtype)
    feat = field.color[alive] * field.opacity[alive].unsqueeze(-1)  # (N, 3) — opacity baked in
    batch = GaussianBatch(
        xy=field.mu[alive],
        scale=torch.exp(field.log_scale[alive]),
        rot=field.rotation[alive],
        feat=feat,
    )
    out = _RASTERIZER(batch, output_hw=output_hw)  # (3, H, W)
    return out.unsqueeze(0)


__all__ = ["render_field"]

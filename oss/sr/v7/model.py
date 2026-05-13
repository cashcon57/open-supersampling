"""V7Model — N-D Gaussian time-slice SR + frame extrapolation.

Composes the v7 ingredients into an end-to-end module:
  backbone(lr)                   -> refined_hr (B, F, H, W)
  canvas.render_at(t_query)      -> canvas_hr  (B, R, H, W)
  composite_head(cat(...))       -> delta
  out = bicubic_hr + delta

Spawning of new Gaussians is decoupled here from the forward pass:
the trainer (Phase 2B) drives `canvas.add(...)` via a separate spawner
module each step. The forward path only consumes whatever the canvas
currently has.

This skeleton uses a placeholder ConvNet backbone (a few residual
blocks) so the integration is verifiable end-to-end without depending
on HAT-Tiny machinery. A subsequent commit swaps the backbone for
HAT-Tiny (or whatever architecture v7-pico-005 picks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.v7.nd_canvas_state import NDCanvasState
from oss.sr.v7.nd_rasterizer import render_nd_time_slice


@dataclass
class V7Config:
    """V7Model configuration. Defaults target pico-tier."""
    in_channels: int = 9        # rgb + depth + motion(2) + normals(3)
    scale: int = 2
    feat_dim: int = 32          # backbone feature width
    latent_rank: int = 16       # canvas feature dim R
    canvas_capacity: int = 4096
    backbone_blocks: int = 4


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.gelu(self.conv1(x))
        h = self.conv2(h)
        return x + h


class _PlaceholderBackbone(nn.Module):
    """Tiny ConvNet that maps LR (B, in_ch, H_lr, W_lr) to HR feature
    map (B, feat_dim, H, W). Bilinear upsample + a few residual blocks.
    Stands in for HAT-Tiny / HAT-L in v7 testing; full backbone swap
    happens later."""

    def __init__(self, in_channels: int, feat_dim: int, scale: int, blocks: int):
        super().__init__()
        self.scale = scale
        self.stem = nn.Conv2d(in_channels, feat_dim, 3, padding=1)
        self.blocks = nn.ModuleList([_ResBlock(feat_dim) for _ in range(blocks)])

    def forward(self, lr_inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(lr_inputs)
        x = F.gelu(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x


class V7Model(nn.Module):
    """End-to-end v7 model: backbone + N-D canvas time-slice + fusion."""

    def __init__(self, cfg: Optional[V7Config] = None):
        super().__init__()
        self.cfg = cfg or V7Config()
        self.backbone = _PlaceholderBackbone(
            in_channels=self.cfg.in_channels,
            feat_dim=self.cfg.feat_dim,
            scale=self.cfg.scale,
            blocks=self.cfg.backbone_blocks,
        )
        self.composite_head = nn.Sequential(
            nn.Conv2d(self.cfg.feat_dim + self.cfg.latent_rank, self.cfg.feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.cfg.feat_dim, max(16, self.cfg.feat_dim // 2), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(16, self.cfg.feat_dim // 2), 3, 3, padding=1),
        )
        # Small-magnitude init on last layer so output starts NEAR
        # bicubic (delta ~= 0) but signal can still flow through the
        # head for backbone + canvas gradients. v6 used pure zero-init
        # which works there because GAN + perceptual losses immediately
        # introduce non-zero gradients elsewhere; v7's training loop
        # (Phase 2B+) will adopt the same setup, but for the skeleton
        # the small-init lets unit tests verify gradient flow without
        # depending on the loss recipe.
        nn.init.normal_(self.composite_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.composite_head[-1].bias)

        # Canvas state holder; lazily allocated on first forward so we
        # know the device.
        self._canvas: Optional[NDCanvasState] = None

    @property
    def canvas(self) -> NDCanvasState:
        if self._canvas is None:
            raise RuntimeError(
                "V7Model.canvas accessed before allocate_canvas() / first forward; "
                "call model.allocate_canvas(device) explicitly or run forward once."
            )
        return self._canvas

    def allocate_canvas(self, device: torch.device | str) -> NDCanvasState:
        """Allocate the canvas pool on the requested device."""
        self._canvas = NDCanvasState.empty(
            capacity=self.cfg.canvas_capacity,
            feature_dim=self.cfg.latent_rank,
            device=device,
            dtype=torch.float32,
        )
        return self._canvas

    def reset_state(self, device: torch.device | str = "cpu") -> None:
        """Clear canvas. Trajectory boundary hook."""
        if self._canvas is None:
            self.allocate_canvas(device)
        else:
            self._canvas.reset()

    def render_canvas(self, t_query: float, output_hw: tuple[int, int]) -> torch.Tensor:
        """Render the canvas at t_query to (1, R, H, W). Empty canvas
        produces a zero tensor."""
        canvas = self.canvas
        if canvas.count == 0:
            return torch.zeros(
                (1, self.cfg.latent_rank, output_hw[0], output_hw[1]),
                device=canvas.device,
                dtype=torch.float32,
            )
        pos, cov, feat, opacity = canvas.active_view()
        rendered = render_nd_time_slice(
            means=pos, covs=cov, features=feat, opacities=opacity,
            t_query=t_query, image_hw=output_hw,
        )
        # Add batch dim
        return rendered.unsqueeze(0)

    def forward(
        self,
        lr_inputs: torch.Tensor,           # (B, in_ch, H_lr, W_lr)
        t_query: float = 0.0,              # absolute time coordinate to render
        output_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """End-to-end forward. Returns (B, 3, H, W) HR image at t = t_query.

        Args:
            lr_inputs: LR + G-buffer stack, B-batched.
            t_query: absolute time coordinate. t = N (integer frame) -> SR
                at current frame. t = N + 0.5 -> OSS-FX intermediate.
            output_hw: HR (H, W). If None, derived from LR shape * scale.
        """
        b, _, h_lr, w_lr = lr_inputs.shape
        if output_hw is None:
            output_hw = (h_lr * self.cfg.scale, w_lr * self.cfg.scale)
        h_hr, w_hr = output_hw

        if self._canvas is None:
            self.allocate_canvas(lr_inputs.device)

        # Backbone -> refined_hr
        refined_hr = self.backbone(lr_inputs)
        # Defensive resize (in case backbone scale mismatches)
        if refined_hr.shape[-2:] != (h_hr, w_hr):
            refined_hr = F.interpolate(refined_hr, size=(h_hr, w_hr),
                                       mode="bilinear", align_corners=False)

        # Canvas -> canvas_hr
        canvas_hr = self.render_canvas(t_query=t_query, output_hw=(h_hr, w_hr))
        canvas_hr = canvas_hr.to(device=refined_hr.device, dtype=refined_hr.dtype)
        if canvas_hr.shape[0] != b:
            canvas_hr = canvas_hr.expand(b, -1, -1, -1)

        # Bicubic anchor (matches v6 design)
        lr_rgb = lr_inputs[:, :3]
        bicubic_hr = F.interpolate(
            lr_rgb, size=(h_hr, w_hr), mode="bicubic", antialias=True, align_corners=False
        ).clamp(min=0.0)

        # Fuse + delta
        delta = self.composite_head(torch.cat([refined_hr, canvas_hr], dim=1))
        return (bicubic_hr + delta).clamp(0.0, 1.0)

"""GaussianTemporalSRModel — full Sprint-5 Gaussian-temporal pipeline.

Wires together:
    - GBufferEncoder         (Task 2)
    - warp_field             (Task 1)  — analytical Gaussian warp
    - GaussianMultiFrameTransformer (Task 3)
    - densify                (Task 4)
    - render_field           (Task 6)  — wrapper around the OSS rasterizer
    - prune                  (Task 5)

Forward signature::

    forward(lr_inputs (B,12,h,w), motion_lr (B,2,h,w), prev_field: GaussianField | None)
        -> (rendered_hr, new_field, debug)

Critical implementation note: after densification we **re-render** the field so
that the first-frame output (and any frame where densification adds Gaussians)
is not the pre-densify zero render. The renderer is differentiable through
``field.color`` so the gradient path stays intact.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.gaussian_temporal.analytical_warp import warp_field
from oss.sr.gaussian_temporal.densification import densify
from oss.sr.gaussian_temporal.g_buffer_encoder import GBufferEncoder
from oss.sr.gaussian_temporal.gaussian_field import GaussianField
from oss.sr.gaussian_temporal.pruning import prune
from oss.sr.gaussian_temporal.rasterizer import render_field
from oss.sr.gaussian_temporal.transformer import GaussianMultiFrameTransformer


class GaussianTemporalSRModel(nn.Module):
    def __init__(self, in_channels: int = 12, scale: int = 2, max_count: int = 16384) -> None:
        super().__init__()
        self.scale = scale
        self.max_count = max_count
        self.encoder = GBufferEncoder(in_channels=in_channels, feat_dim=128, tile_size=16)
        self.transformer = GaussianMultiFrameTransformer(
            d_model=128, n_heads=4, n_layers=4, history_len=5,
        )
        # Class-level constants required by the plan.
        self.initial_seed_count = 4096
        self.densify_threshold = 0.05
        self.densify_max_new = 256
        self.opacity_threshold = 0.05

    def forward(
        self,
        lr_inputs: torch.Tensor,
        motion_lr: torch.Tensor,
        prev_field: Optional[GaussianField],
    ) -> tuple[torch.Tensor, GaussianField, dict]:
        b, _, h_lr, w_lr = lr_inputs.shape
        h_hr, w_hr = h_lr * self.scale, w_lr * self.scale
        if b != 1:
            raise ValueError(f"GaussianTemporalSRModel expects B=1; got {b}.")

        feats = self.encoder(lr_inputs)               # (1, 128, h/16, w/16)

        # ---- First-frame seed -------------------------------------------------
        if prev_field is None:
            # Empty field; seed via densification so count_alive > 0 at frame 0.
            # Target = bilinear-upscale of LR RGB; baseline rendered = zeros.
            warped = GaussianField(capacity=self.max_count, device=lr_inputs.device)
            lr_rgb = lr_inputs[:, :3]
            target_hr = F.interpolate(
                lr_rgb, size=(h_hr, w_hr), mode="bilinear", align_corners=False
            )
            zero_render = torch.zeros_like(target_hr)
            warped = densify(
                warped,
                lr_target=target_hr,
                rendered=zero_render,
                tile_size=self.scale * 16,  # match encoder tile size at HR
                residual_threshold=0.0,
                max_new=self.initial_seed_count,
            )
            history: list[GaussianField] = []
        else:
            warped = warp_field(prev_field, motion_lr[0], hw=(h_lr, w_lr))
            history = prev_field.history

        # ---- Transformer update over alive tokens -----------------------------
        if warped.count_alive() > 0:
            updates = self.transformer(field_curr=warped, history=history, tile_features=feats)
            alive_idx = warped.alive.nonzero(as_tuple=True)[0]
            warped.mu = warped.mu.clone()
            warped.log_scale = warped.log_scale.clone()
            warped.rotation = warped.rotation.clone()
            warped.color = warped.color.clone()
            warped.mu[alive_idx] = warped.mu[alive_idx] + updates["dmu"]
            warped.log_scale[alive_idx] = warped.log_scale[alive_idx] + updates["dlog_scale"]
            warped.rotation[alive_idx] = warped.rotation[alive_idx] + updates["drot"]
            warped.color[alive_idx] = warped.color[alive_idx] + updates["dcolor"]

        # ---- First render -----------------------------------------------------
        rendered_hr = render_field(warped, output_hw=(h_hr, w_hr))

        # ---- Densify on residual vs LR-upsampled-target -----------------------
        # Match Phase 1+2+3 spec — residual densification active in the model.
        # Train loop can also do an additional pass against GT HR if desired.
        lr_rgb = lr_inputs[:, :3]
        target_hr = F.interpolate(
            lr_rgb, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )
        warped = densify(
            warped,
            lr_target=target_hr,
            rendered=rendered_hr,
            tile_size=self.scale * 16,
            residual_threshold=self.densify_threshold,
            max_new=self.densify_max_new,
        )

        # ---- Re-render after densification so frame-0 (and any frame where
        # densification adds Gaussians) does NOT return the pre-densify image.
        # Gradient still flows: render_field is differentiable through field.color.
        rendered_hr = render_field(warped, output_hw=(h_hr, w_hr))

        # ---- Prune ------------------------------------------------------------
        new_field = prune(
            warped, opacity_threshold=self.opacity_threshold, max_count=self.max_count
        )

        debug = {"count_alive": int(new_field.count_alive())}
        return rendered_hr, new_field, debug


__all__ = ["GaussianTemporalSRModel"]

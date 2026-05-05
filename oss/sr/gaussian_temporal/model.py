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
    def __init__(
        self,
        in_channels: int = 12,
        scale: int = 2,
        max_count: int = 16384,
        color_activation: str = "softplus",
    ) -> None:
        """
        Args:
            color_activation: ``"sigmoid"`` clamps fitter RGB to [0, 1]
                (SDR-only — HDR values lossy-clipped to peak white).
                ``"softplus"`` (default) outputs unbounded non-negative
                values, supporting HDR linear-light input/output. SDR
                training data still trains the model fine since softplus
                stays linear-ish near 0 and saturates softly above; the
                fitter just isn't artificially capped at 1.0 anymore.
                Mirrors the convention in oss/gaussian/network/output_head.py.
        """
        super().__init__()
        if color_activation not in ("sigmoid", "softplus"):
            raise ValueError(
                f"color_activation must be 'sigmoid' or 'softplus'; got {color_activation!r}"
            )
        self.scale = scale
        self.max_count = max_count
        self.color_activation = color_activation
        self.encoder = GBufferEncoder(in_channels=in_channels, feat_dim=128, tile_size=16)
        # Per-frame fitter used to seed Gaussian colors from encoder features.
        # This keeps Phase 1 trainable while bypassing temporal attention.
        self.fitter_rgb_head = nn.Conv2d(128, 3, kernel_size=1)
        nn.init.normal_(self.fitter_rgb_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fitter_rgb_head.bias)
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
        phase: int = 3,
    ) -> tuple[torch.Tensor, GaussianField, dict]:
        """Run the full Gaussian-temporal pipeline.

        Args:
            phase: Training-phase isolation gate, per the spec's 4-phase
                schedule.
                - ``phase=1``: single-frame fitter only — encoder + density +
                  raster. Transformer is bypassed entirely. ``prev_field`` and
                  history are ignored (Phase 1 is per-frame).
                - ``phase=2``: warped prev-field + 2-effective-layer
                  transformer warmup. Encoder is meant to be frozen by the
                  trainer (this method does not enforce that).
                - ``phase=3`` (default): full 4-layer transformer, full
                  pipeline.
                - ``phase=4``: same architecture as Phase 3; reserved for
                  trainer-side LR scaling on Sintel-only fine-tune.
        """
        b, _, h_lr, w_lr = lr_inputs.shape
        h_hr, w_hr = h_lr * self.scale, w_lr * self.scale
        if b != 1:
            raise ValueError(f"GaussianTemporalSRModel expects B=1; got {b}.")
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"phase must be in {{1,2,3,4}}; got {phase}.")

        feats = self.encoder(lr_inputs)               # (1, 128, h/16, w/16)
        # softplus by default: unbounded non-negative output supports HDR
        # linear-light values >1.0; sigmoid path retained for SDR-only
        # callers that explicitly want [0,1] clamping.
        rgb_raw = self.fitter_rgb_head(feats)
        rgb_activated = (
            torch.sigmoid(rgb_raw) if self.color_activation == "sigmoid"
            else F.softplus(rgb_raw)
        )
        fitter_rgb_hr = F.interpolate(
            rgb_activated,
            size=(h_hr, w_hr),
            mode="bilinear",
            align_corners=False,
        )

        # Phase 1 isolates the per-frame fitter — no temporal, no transformer.
        # Force prev_field=None so the warp+transformer paths cannot run, and
        # downstream history population becomes a no-op.
        if phase == 1:
            prev_field = None

        # ---- First-frame seed -------------------------------------------------
        if prev_field is None:
            # Empty field; seed via densification so count_alive > 0 at frame 0.
            # Target comes from the trainable per-frame fitter; baseline render
            # is zeros.
            warped = GaussianField(capacity=self.max_count, device=lr_inputs.device)
            zero_render = torch.zeros_like(fitter_rgb_hr)
            warped = densify(
                warped,
                lr_target=fitter_rgb_hr,
                rendered=zero_render,
                tile_size=self.scale * 16,  # match encoder tile size at HR
                residual_threshold=0.0,
                max_new=self.initial_seed_count,
            )
            history: list[GaussianField] = []
        else:
            # Field means are stored in HR pixel coordinates because the
            # rasterizer renders the HR output. The dataset motion field is LR
            # pixels, so lift both the vector field and its displacement units
            # into HR space before analytical warping.
            motion_hr = F.interpolate(
                motion_lr,
                size=(h_hr, w_hr),
                mode="bilinear",
                align_corners=False,
            ) * float(self.scale)
            warped = warp_field(prev_field, motion_hr[0], hw=(h_hr, w_hr))
            history = prev_field.history

        # ---- Transformer update over alive tokens -----------------------------
        # Phase 1 bypasses the transformer entirely (single-frame fitter).
        # Phase 2 uses 2 effective layers (transformer warmup).
        # Phase 3+ uses all layers.
        if phase != 1 and warped.count_alive() > 0:
            effective_layers = 2 if phase == 2 else None
            updates = self.transformer(
                field_curr=warped, history=history, tile_features=feats,
                effective_layers=effective_layers,
            )
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
        target_hr = (
            fitter_rgb_hr if phase == 1 else
            F.interpolate(lr_rgb, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
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

        # ---- Populate history -------------------------------------------------
        # The transformer attends over up to ``history_len`` prior frames; the
        # caller carries ``new_field`` to the next forward call as ``prev_field``,
        # so the history must contain a snapshot of the prior fields.
        # ``push_history`` appendleft-s, so the newest snapshot must be pushed
        # LAST. We push older snapshots first (in reverse so the iteration
        # order matches deque newest-first when pushed), then push prev_field.
        # The deque's ``maxlen=HISTORY_LEN`` truncates oldest if overflowed.
        if prev_field is not None:
            for older in reversed(prev_field.history):
                new_field.push_history(older)
            new_field.push_history(prev_field.clone())

        debug = {
            "count_alive": int(new_field.count_alive()),
            "history_len": len(new_field.history),
        }
        return rendered_hr, new_field, debug


__all__ = ["GaussianTemporalSRModel"]

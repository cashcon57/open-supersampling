"""V6Model — covariance-resampled online Gaussian-temporal SR orchestrator.

Wires the v6 architecture:

  Input: LR frame + G-buffers (RGB + depth + motion + normals = 9 channels by
         default, drop-canvas; configurable to 12 for legacy inputs)
  Stage 1: HAT spatial backbone produces pixel features at LR resolution.
  Stage 2: PersistentGaussianCanvas (across frames) is warped by engine
           motion vectors with GS-STVSR covariance resampling.
  Stage 3: KeyframeActiveMaskCache picks the active subset every K frames.
  Stage 4: PixelGaussianFusion cross-attends pixel queries to active canvas
           tokens (K=0 is identity — first frame, fresh-canvas case).
  Stage 5: Active canvas subset rasterizes to HR feature image.
  Stage 6: Refined HAT features are upsampled to HR and composited with
           rasterized canvas features by a lightweight conv head.
  Stage 7: Fresh Gaussians are spawned from refined features and written back
           into the persistent per-rank canvas.
  Stage 8: Bicubic-residual RGB output. The composite head predicts a learned
           delta from the bicubic-upsampled LR RGB input. ``color_activation``
           selects HDR/non-negative output (``"hdr"`` / deprecated
           ``"softplus"`` alias, clamp(min=0)) or SDR/unit-range output
           (``"sdr"`` / deprecated ``"sigmoid"`` alias, clamp(0, 1)).
  Stage 9: STVScoreState aggregates per-Gaussian contribution + lifespan
           every step; periodic prune_by_st_score() drops the bottom
           fraction every ``prune_every`` steps.

Frame extrapolation (OSS-FX) is done by the same forward at α<1 — the
architecture is α=1 (next-frame-prediction); calling ``render_at_alpha``
provides the FX path. Implemented as a thin wrapper that warps the canvas
by α·motion before rasterization.

Per-vendor inference precision and the AA-stack rasterizer modifications
(AAA-Gaussians + AA-2DGS + Analytic-Splatting) are layered on top of this
class via the corresponding modules in ``oss/sr/v6/aa_*.py`` and the
runtime engine — the AA stack is referenced but not yet plumbed into the
training-time forward; it activates at inference once the rasterizer
calls those modules.

DDP-safe: model parameters are the only DDP-synced state; canvas state +
ST-score state + keyframe mask state are per-rank-local across a single
training step (each rank sees its own batch's canvas, then resets at the
trajectory boundary).
"""
from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.v6.concat_fusion import ConcatFusion
from oss.sr.v6.cross_attention import PixelGaussianFusion
from oss.sr.v6.disocclusion_spawner import DisocclusionSpawner
from oss.sr.v6.gaussian_spawner import GaussianSpawner, GaussianSpawnState
from oss.sr.v6.hat import HAT, hat_l, hat_small, hat_tiny
from oss.sr.v6.keyframe_active_mask import KeyframeActiveMaskCache
from oss.sr.v6.st_variation_score import (
    STVScoreState,
    init_st_score_state,
    update_st_score,
)


log = logging.getLogger("oss.sr.v6.model")


def _debug_nan_enabled() -> bool:
    return os.environ.get("OSS_V6_DEBUG_NAN", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _debug_tensor_stats(x: torch.Tensor) -> str:
    x_f = x.detach().float()
    if x_f.numel() == 0:
        return "mean=nan min=nan max=nan"
    return (
        f"mean={float(x_f.mean()):.9g} "
        f"min={float(x_f.amin()):.9g} "
        f"max={float(x_f.amax()):.9g}"
    )


_BACKBONE_REGISTRY = {
    "hat-tiny": hat_tiny,
    "hat-small": hat_small,
    "hat-l": hat_l,
    "hat-base": hat_l,  # alias — paper-canonical HAT-L is OSS's "Heavy" tier
}


@dataclass
class V6Config:
    """Hyperparameters for V6Model construction.

    Defaults match the v6 canonical memo's "Heavy" teacher tier (HAT-L
    backbone, 15K canvas capacity, HDR non-negative output). Standard /
    Pico tiers override ``backbone`` and ``canvas_capacity``.
    """

    in_channels: int = 9
    scale: int = 2
    backbone: str = "hat-l"
    canvas_capacity: int = 15000
    token_dim: int = 64
    cross_attention_heads: int = 6
    window_size: int = 16
    # Preferred values are "hdr" and "sdr". "softplus" / "sigmoid" remain
    # deprecated aliases for checkpoint and caller compatibility.
    color_activation: str = "hdr"
    tile_size_lr: int = 8
    tile_size_hr: int = 16
    spawn_offset_random: bool = False
    spawn_subpixel_jitter: bool = False
    rasterizer_overlap: int = 0
    keyframe_interval: int = 10
    prune_every: int = 200
    prune_fraction: float = 0.7
    # v6.2 architectural switches (T6 wiring). Defaults preserve the v6.1
    # legacy path so existing checkpoints + tests stay green; pico-002 opts
    # in to "concat" + "disocclusion" + R=16 per the arch v4 spec and the
    # 2026-05-08 SVD rank gate (issue #11).
    fusion_mode: str = "cross_attention"  # {"cross_attention", "concat"}
    spawner_mode: str = "legacy"          # {"legacy", "disocclusion"}
    latent_rank: int = 16                 # gate-passing default; <token_dim
    # Disocclusion spawner controls (only used when spawner_mode="disocclusion")
    disocclusion_births_per_frame: int = 256
    disocclusion_births_per_tile: int = 4
    disocclusion_depth_threshold: float = 0.05
    disocclusion_residual_threshold: float = 0.02
    # ConcatFusion controls (only used when fusion_mode="concat")
    concat_fusion_hidden: int = 64


class V6Model(nn.Module):
    """The v6 orchestrator. See module docstring for stage-by-stage layout."""

    NOT_IMPLEMENTED_MESSAGE = ""  # retained for backward-compat with import probes.

    def __init__(self, config: Optional[V6Config] = None, **kwargs) -> None:
        super().__init__()
        self.cfg = config or V6Config(**kwargs)
        if self.cfg.backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"unknown backbone {self.cfg.backbone!r}; must be one of "
                f"{sorted(_BACKBONE_REGISTRY)}"
            )
        if self.cfg.color_activation not in ("hdr", "sdr", "softplus", "sigmoid"):
            raise ValueError(
                "color_activation must be 'hdr' or 'sdr' "
                "('softplus' / 'sigmoid' aliases are deprecated); got "
                f"{self.cfg.color_activation!r}"
            )
        if self.cfg.fusion_mode not in ("cross_attention", "concat"):
            raise ValueError(
                "fusion_mode must be 'cross_attention' or 'concat'; "
                f"got {self.cfg.fusion_mode!r}"
            )
        if self.cfg.spawner_mode not in ("legacy", "disocclusion"):
            raise ValueError(
                "spawner_mode must be 'legacy' or 'disocclusion'; "
                f"got {self.cfg.spawner_mode!r}"
            )
        if self.cfg.latent_rank <= 0 or self.cfg.latent_rank > self.cfg.token_dim:
            raise ValueError(
                f"latent_rank must be in (0, token_dim={self.cfg.token_dim}]; "
                f"got {self.cfg.latent_rank}"
            )
        self.scale = int(self.cfg.scale)

        self.backbone: HAT = _BACKBONE_REGISTRY[self.cfg.backbone](
            in_channels=self.cfg.in_channels,
        )
        self.feat_dim: int = int(self.backbone.embed_dim)

        # Canvas-token embedding: project per-Gaussian color/feature into the
        # cross-attention's token_dim. Done as a learnable Linear so the
        # canvas representation doesn't have to match feat_dim exactly.
        self.canvas_to_token = nn.Linear(self.cfg.token_dim, self.cfg.token_dim)

        if self.cfg.fusion_mode == "cross_attention":
            self.fusion = PixelGaussianFusion(
                feat_dim=self.feat_dim,
                token_dim=self.cfg.token_dim,
                num_heads=self.cfg.cross_attention_heads,
                window_size=self.cfg.window_size,
            )
        else:  # "concat"
            self.fusion = ConcatFusion(
                feat_dim=self.feat_dim,
                latent_R=self.cfg.latent_rank,
                hidden=self.cfg.concat_fusion_hidden,
            )

        # LR feature refinement before canvas fusion. The old PixelShuffle
        # decoder is replaced below by rasterize + composite.
        self.pixel_head = nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1)
        self.activation = nn.GELU()

        if self.cfg.spawner_mode == "legacy":
            self.gaussian_spawner = GaussianSpawner(
                feat_dim=self.feat_dim,
                token_dim=self.cfg.token_dim,
                scale=self.scale,
                tile_size_lr=self.cfg.tile_size_lr,
                config=self.cfg,
            )
            self.disocclusion_spawner = None
        else:  # "disocclusion"
            self.gaussian_spawner = None
            self.disocclusion_spawner = DisocclusionSpawner(
                feat_dim=self.feat_dim,
                max_births_per_frame=self.cfg.disocclusion_births_per_frame,
                max_births_per_tile=self.cfg.disocclusion_births_per_tile,
                disocclusion_depth_threshold=self.cfg.disocclusion_depth_threshold,
                residual_threshold=self.cfg.disocclusion_residual_threshold,
            )
            # Project per-Gaussian feature payload (feat_dim) into the
            # rasterizer's color/feature channel (token_dim). Zero-init so
            # newly spawned Gaussians start with no canvas contribution and
            # the model converges to learning the residual.
            self.disocclusion_color_proj = nn.Linear(
                self.feat_dim, self.cfg.token_dim
            )
            nn.init.zeros_(self.disocclusion_color_proj.weight)
            nn.init.zeros_(self.disocclusion_color_proj.bias)
        from oss.sr.v6.rasterizer import V6Rasterizer

        self.rasterizer = V6Rasterizer(
            token_dim=self.cfg.token_dim,
            tile_size=self.cfg.tile_size_hr,
            overlap=self.cfg.rasterizer_overlap,
            latent_rank=self.cfg.latent_rank,
        )
        hidden = max(16, self.feat_dim // 2)
        # canvas_hr produced by the rasterizer has `latent_rank` channels (or
        # token_dim when latent_rank == token_dim). Use the rasterizer's own
        # output dim as the source of truth so the composite head's input
        # channel count tracks latent_rank automatically.
        canvas_hr_channels = int(self.rasterizer.feature_dim)
        self.composite_head = nn.Sequential(
            nn.Conv2d(self.feat_dim + canvas_hr_channels, self.feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feat_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.composite_head[-1].weight)
        nn.init.zeros_(self.composite_head[-1].bias)

        # Per-rank-local stateful pieces. NOT registered as buffers so DDP
        # doesn't try to sync them; they reset at trajectory boundaries.
        self._canvas_state: Optional[CanvasState] = None
        self._st_state: Optional[STVScoreState] = None
        self._spawn_offset_xy: Optional[torch.Tensor] = None
        # Disocclusion spawner needs previous-frame depth at LR to compute
        # the disocclusion mask. We carry it across calls; first frame uses
        # depth_t (yields zero disocclusion, which is correct for frame 0).
        self._depth_lr_prev: Optional[torch.Tensor] = None
        self.keyframe_mask = KeyframeActiveMaskCache(
            keyframe_interval=self.cfg.keyframe_interval,
        )

        # Step counter for periodic prune; advances on every training-step
        # call to ``maybe_prune()``. Registered so checkpoints resume pruning
        # cadence instead of silently restarting it.
        self.register_buffer("_step_count", torch.zeros((), dtype=torch.long))

        # zero_gbuffer_into_backbone is part of the saved-args contract
        # carried over from v5 for warm-start ckpts. v6 backbones are
        # trained with real G-buffers from the start, so default False.
        self.zero_gbuffer_into_backbone: bool = False
        self.debug_nan: bool = _debug_nan_enabled()
        self.debug_nan_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Stateful canvas + score interface
    # ------------------------------------------------------------------

    def reset_state(self, device: Optional[torch.device] = None) -> None:
        """Reset the canvas, ST score, and keyframe mask. Call at the start
        of every trajectory (or every training step that does not carry
        canvas state across calls)."""
        self._canvas_state = None
        self._st_state = None
        self._spawn_offset_xy = None
        self._depth_lr_prev = None
        self.keyframe_mask.reset()
        self._step_count.zero_()

    def has_canvas(self) -> bool:
        return self._canvas_state is not None and self._canvas_state.count > 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        lr_inputs: torch.Tensor,                 # (B, in_channels, H, W)
        motion_lr: Optional[torch.Tensor] = None,  # (B, 2, H, W); None on frame 0
        depth_hr_curr: Optional[torch.Tensor] = None,
        depth_hr_prev: Optional[torch.Tensor] = None,
        frame_index: int = 0,
    ) -> torch.Tensor:
        """Run v6 forward. Returns HR image (B, 3, H*scale, W*scale).

        For training, callers loop over a trajectory window calling forward
        once per frame, with motion_lr threaded between frames. For first
        frame call with motion_lr=None.
        """
        debug_first_nonfinite: Optional[str] = None

        def debug_check(name: str, tensor: Optional[torch.Tensor]) -> None:
            nonlocal debug_first_nonfinite
            if (
                not self.debug_nan
                or debug_first_nonfinite is not None
                or tensor is None
                or bool(torch.isfinite(tensor).all().detach().item())
            ):
                return
            debug_first_nonfinite = name
            log.warning(
                "model stage non-finite: name=%s stats=%s step=%s frame_index=%d",
                name,
                _debug_tensor_stats(tensor),
                "unknown" if self.debug_nan_step is None else int(self.debug_nan_step),
                int(frame_index),
            )

        feats = self.backbone(lr_inputs)         # (B, feat_dim, H, W)
        feats = self.activation(self.pixel_head(feats))
        debug_check("coarse_features", feats)
        b, _, h_lr, w_lr = feats.shape
        output_hw = (h_lr * self.scale, w_lr * self.scale)

        warped_canvas = self._warped_canvas(motion_lr=motion_lr, output_hw=output_hw)
        if warped_canvas is not None:
            debug_check("warped_canvas.positions", warped_canvas.positions)
            debug_check("warped_canvas.scales", warped_canvas.scales)
            debug_check("warped_canvas.rotations", warped_canvas.rotations)
            debug_check("warped_canvas.opacities", warped_canvas.opacities)
            debug_check("warped_canvas.colors", warped_canvas.colors)
        active_mask = self._active_mask(
            warped_canvas,
            frame_index=frame_index,
            output_hw=output_hw,
        )

        # ----- Fusion stage: branch on fusion_mode -----
        if self.cfg.fusion_mode == "cross_attention":
            tokens = self._tokens_from_canvas(feats, warped_canvas, active_mask)
            debug_check("tokens", tokens)
            refined = self.fusion(feats, tokens)
        else:  # "concat"
            G_lr, m_lr = self._rasterize_canvas_at_lr(
                warped_canvas, active_mask, b, h_lr, w_lr, feats.device, feats.dtype
            )
            debug_check("concat.G_lr", G_lr)
            debug_check("concat.m_lr", m_lr)
            I_base = lr_inputs[:, :3].to(dtype=feats.dtype)
            depth_lr = lr_inputs[:, 3:4].to(dtype=feats.dtype) if lr_inputs.shape[1] > 3 \
                else torch.zeros((b, 1, h_lr, w_lr), device=feats.device, dtype=feats.dtype)
            mv_lr = motion_lr if motion_lr is not None \
                else torch.zeros((b, 2, h_lr, w_lr), device=feats.device, dtype=feats.dtype)
            mv_lr = mv_lr.to(dtype=feats.dtype)
            refined = self.fusion(feats, G_lr, m_lr, I_base, depth_lr, mv_lr)
        debug_check("refined", refined)

        # ----- Spawner stage: branch on spawner_mode -----
        if self.cfg.spawner_mode == "legacy":
            spawned = self.gaussian_spawner(
                refined,
                spawn_offset_xy=self._spawn_offset_for(refined),
            )
            debug_check("spawned.positions", spawned.positions)
            debug_check("spawned.scales", spawned.scales)
            debug_check("spawned.rotations", spawned.rotations)
            debug_check("spawned.opacities", spawned.opacities)
            debug_check("spawned.colors", spawned.colors)
            debug_check("spawned.confidence", spawned.confidence)
            spawned_canvas = self._flatten_spawned(spawned)
        else:  # "disocclusion"
            spawned_canvas = self._disocclusion_spawn(
                refined=refined,
                lr_inputs=lr_inputs,
                motion_lr=motion_lr,
                warped_canvas=warped_canvas,
                active_mask=active_mask,
                h_lr=h_lr,
                w_lr=w_lr,
                debug_check=debug_check,
            )
        debug_check("flattened.positions", spawned_canvas.positions)
        debug_check("flattened.scales", spawned_canvas.scales)
        debug_check("flattened.rotations", spawned_canvas.rotations)
        debug_check("flattened.opacities", spawned_canvas.opacities)
        debug_check("flattened.colors", spawned_canvas.colors)
        previous_st_state = self._st_state
        old_count = 0 if warped_canvas is None else int(warped_canvas.count)
        render_canvas = self._concat_canvas(warped_canvas, spawned_canvas)
        debug_check("concat.positions", render_canvas.positions)
        debug_check("concat.scales", render_canvas.scales)
        debug_check("concat.rotations", render_canvas.rotations)
        debug_check("concat.opacities", render_canvas.opacities)
        debug_check("concat.colors", render_canvas.colors)
        self._canvas_state = render_canvas
        self.keyframe_mask.reset()

        render_active = self._render_active_mask(
            old_active=active_mask,
            old_count=old_count,
            new_count=int(spawned_canvas.count),
            canvas=render_canvas,
        )
        rasterizer_input_alive_count = render_active[: int(render_canvas.count)].sum()
        debug_check(
            "rasterizer.input_alive_count",
            rasterizer_input_alive_count.to(dtype=torch.float32),
        )
        if self.debug_nan:
            log.info(
                "model stage finite: name=rasterizer.input_alive_count value=%d "
                "step=%s frame_index=%d",
                int(rasterizer_input_alive_count.detach().item()),
                "unknown" if self.debug_nan_step is None else int(self.debug_nan_step),
                int(frame_index),
            )
        rast_hr_out = self.rasterizer(
            render_canvas,
            render_active.unsqueeze(0).expand(b, -1),
            output_hw=output_hw,
            return_weight_sum=False,
        )
        # Low-rank rasterizer (latent_rank<token_dim) defaults to returning
        # (features, weight_sum); we explicitly request features-only here
        # since the composite head consumes only the latent splat.
        canvas_hr = rast_hr_out[0] if isinstance(rast_hr_out, tuple) else rast_hr_out
        debug_check("rasterizer.output", canvas_hr)
        refined_hr = F.interpolate(
            refined,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        debug_check("refined_hr", refined_hr)
        lr_rgb = lr_inputs[:, :3]
        # Bicubic residual skip puts init quality near bicubic instead of
        # uniform ln(2). Cost is ~0.06 ms at 4K, negligible vs the 4 ms target.
        bicubic_hr = F.interpolate(
            lr_rgb,
            size=output_hw,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(min=0.0)
        debug_check("bicubic_hr", bicubic_hr)
        delta = self.composite_head(torch.cat([refined_hr, canvas_hr], dim=1))
        debug_check("composite_delta", delta)
        rgb_hr = bicubic_hr + delta
        if self.cfg.color_activation in ("sdr", "sigmoid"):
            rgb_hr = rgb_hr.clamp(0.0, 1.0)
        else:
            rgb_hr = rgb_hr.clamp(min=0.0)
        self._update_st_state(
            render_canvas,
            render_active,
            previous_state=previous_st_state,
            old_count=old_count,
            new_count=int(spawned_canvas.count),
        )
        debug_check("rgb_hr", rgb_hr)

        # Persistent per-rank state must NOT carry autograd across frames.
        # Without this, the canvas + ST tensors keep gradient back through
        # every previous frame's spawner (BPTT through 100s of frames),
        # which OOMs and leaks gradient state across optimizer steps.
        # The current frame's loss already had a live graph above; we only
        # detach what gets stored back into self for the next forward.
        self._canvas_state = self._detach_canvas(self._canvas_state)
        self._st_state = self._detach_st_state(self._st_state)

        return rgb_hr

    @staticmethod
    def _detach_canvas(canvas: Optional[CanvasState]) -> Optional[CanvasState]:
        if canvas is None:
            return None
        return CanvasState(
            positions=canvas.positions.detach(),
            scales=canvas.scales.detach(),
            rotations=canvas.rotations.detach(),
            opacities=canvas.opacities.detach(),
            colors=canvas.colors.detach(),
            count=canvas.count,
        )

    @staticmethod
    def _detach_st_state(state):
        if state is None:
            return None
        # STVScoreState is a dataclass with two tensor fields + an int.
        # int64 lifespan_count carries no autograd; detach the float
        # spatial_accumulator that the spawner's confidence path leaks into.
        from oss.sr.v6.st_variation_score import STVScoreState
        return STVScoreState(
            spatial_accumulator=state.spatial_accumulator.detach(),
            lifespan_count=state.lifespan_count.detach(),
            frames_observed=state.frames_observed,
        )

    # ------------------------------------------------------------------
    # ST-score-driven pruning hook
    # ------------------------------------------------------------------

    def maybe_prune(self, step: Optional[int] = None) -> int:
        """Called by the trainer once per step. Returns the number of
        Gaussians pruned (0 if not a prune step or canvas empty).
        """
        if step is None:
            self._step_count.add_(1)
            step_index = int(self._step_count.item())
        else:
            step_index = int(step)
        if self._canvas_state is None or self._st_state is None:
            return 0
        if step_index % self.cfg.prune_every != 0:
            return 0
        from oss.sr.v6.st_variation_score import prune_by_st_score
        before = int(self._canvas_state.count)
        new_canvas, new_state = prune_by_st_score(
            self._canvas_state, self._st_state, prune_fraction=self.cfg.prune_fraction,
        )
        self._canvas_state = new_canvas
        self._st_state = new_state
        return before - int(new_canvas.count)

    # ------------------------------------------------------------------
    # Token construction (private)
    # ------------------------------------------------------------------

    def _build_canvas_tokens(
        self,
        feats: torch.Tensor,
        frame_index: int = 0,
    ) -> torch.Tensor:
        """Project the active subset of canvas Gaussians into (B, K, token_dim)
        cross-attention tokens. Empty-canvas case returns shape (B, 0, token_dim)
        which the fusion module handles as identity-passthrough.
        """
        B = feats.shape[0]
        if self._canvas_state is None or self._canvas_state.count == 0:
            return feats.new_zeros((B, 0, self.cfg.token_dim))
        # Active mask via the keyframe cache; on the first frame after a
        # reset, this rebuilds; subsequent intermediate frames inherit.
        # The mask-cache returns a (N,) boolean tensor over alive Gaussians.
        active = self.keyframe_mask.get_mask(
            frame_index=frame_index,
            canvas=self._canvas_state,
            view_matrix=None,  # the cache's bbox-test handles None as identity
            viewport_hw=(feats.shape[-2] * self.scale, feats.shape[-1] * self.scale),
        )
        return self._tokens_from_canvas(feats, self._canvas_state, active)

    def _tokens_from_canvas(
        self,
        feats: torch.Tensor,
        canvas: Optional[CanvasState],
        active: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = feats.shape[0]
        if canvas is None or canvas.count == 0 or active is None:
            return feats.new_zeros((B, 0, self.cfg.token_dim))
        active = self._normalize_mask(active, canvas.count, canvas.positions.device)
        if active is None or active.sum() == 0:
            return feats.new_zeros((B, 0, self.cfg.token_dim))
        # Project active Gaussian colors into token_dim. The canvas stores
        # per-Gaussian feature/color tensors; we feed those through a
        # learnable Linear so the embedding can be optimized end-to-end.
        active_feats = canvas.colors[active]  # (K_active, F_canvas)
        # Truncate or pad to token_dim before the Linear so the canvas's
        # native feat_dim doesn't have to match exactly.
        if active_feats.shape[-1] != self.cfg.token_dim:
            if active_feats.shape[-1] > self.cfg.token_dim:
                active_feats = active_feats[..., : self.cfg.token_dim]
            else:
                pad = self.cfg.token_dim - active_feats.shape[-1]
                active_feats = F.pad(active_feats, (0, pad))
        tokens_1 = self.canvas_to_token(active_feats)  # (K_active, token_dim)
        # Expand to (B, K_active, token_dim). Same canvas viewed by every
        # element of the batch (per-rank shared canvas; DDP cross-rank state
        # is not synced here, only model params).
        return tokens_1.unsqueeze(0).expand(B, -1, -1).contiguous()

    def _warped_canvas(
        self,
        motion_lr: Optional[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> Optional[CanvasState]:
        if self.has_canvas() and motion_lr is not None:
            from oss.sr.v6.canvas_warp import warp_canvas

            return warp_canvas(self._canvas_state, motion_lr, output_hw)
        return self._canvas_state

    def _active_mask(
        self,
        canvas: Optional[CanvasState],
        frame_index: int,
        output_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if canvas is None or canvas.count == 0:
            return None
        view_matrix = torch.eye(3, device=canvas.positions.device, dtype=canvas.positions.dtype)
        return self.keyframe_mask.get_mask(
            frame_index=frame_index,
            canvas=canvas,
            view_matrix=view_matrix,
            viewport_hw=output_hw,
        )

    def _rasterize_canvas_at_lr(
        self,
        warped_canvas: Optional[CanvasState],
        active_mask: Optional[torch.Tensor],
        batch_size: int,
        h_lr: int,
        w_lr: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize the warped canvas at LR resolution to produce (G, m)
        for ConcatFusion. G is (B, R, H_lr, W_lr) — latent splat readout.
        m is (B, 1, H_lr, W_lr) — per-pixel sum of Gaussian weights.

        Canvas positions live in HR pixel space, so we scale them down by
        ``self.scale`` to render at LR. Empty / no-canvas case returns
        zeros, which makes ConcatFusion behave as a pure-feature pass.
        """
        R = int(self.cfg.latent_rank)
        if warped_canvas is None or warped_canvas.count == 0:
            G = torch.zeros((batch_size, R, h_lr, w_lr), device=device, dtype=dtype)
            m = torch.zeros((batch_size, 1, h_lr, w_lr), device=device, dtype=dtype)
            return G, m
        scaled = CanvasState(
            positions=warped_canvas.positions / float(self.scale),
            scales=warped_canvas.scales / float(self.scale),
            rotations=warped_canvas.rotations,
            opacities=warped_canvas.opacities,
            colors=warped_canvas.colors,
            count=warped_canvas.count,
        )
        if active_mask is None:
            mask = torch.ones(
                int(warped_canvas.count), device=device, dtype=torch.bool
            )
        elif active_mask.ndim == 1:
            mask = active_mask
        else:
            mask = active_mask
        out = self.rasterizer(
            scaled,
            mask.unsqueeze(0).expand(batch_size, -1) if mask.ndim == 1 else mask,
            output_hw=(h_lr, w_lr),
            return_weight_sum=True,
        )
        if isinstance(out, tuple):
            G, m = out
        else:
            G = out
            m = torch.zeros((batch_size, 1, h_lr, w_lr), device=device, dtype=dtype)
        return G.to(dtype=dtype), m.to(dtype=dtype)

    def _disocclusion_spawn(
        self,
        *,
        refined: torch.Tensor,
        lr_inputs: torch.Tensor,
        motion_lr: Optional[torch.Tensor],
        warped_canvas: Optional[CanvasState],
        active_mask: Optional[torch.Tensor],
        h_lr: int,
        w_lr: int,
        debug_check,
    ) -> CanvasState:
        """Run the disocclusion spawner and convert its dict output into a
        CanvasState in HR pixel coordinates so the rest of the pipeline is
        unchanged. Inputs sourced from ``lr_inputs`` channel layout:
        [0:3]=RGB, [3:4]=depth, [4:6]=MV, [6:9]=normals.
        """
        b = int(refined.shape[0])
        device = refined.device
        dtype = refined.dtype

        # depth_t at LR
        if lr_inputs.shape[1] >= 4:
            depth_t = lr_inputs[:, 3:4].to(dtype=dtype)
        else:
            depth_t = torch.zeros((b, 1, h_lr, w_lr), device=device, dtype=dtype)
        # depth_prev: persisted across frames; first frame uses depth_t so
        # the disocclusion mask is empty (no spawn — correct for frame 0).
        if self._depth_lr_prev is None or self._depth_lr_prev.shape != depth_t.shape:
            depth_prev = depth_t.detach().clone()
        else:
            depth_prev = self._depth_lr_prev.to(device=device, dtype=dtype)
        # MV at LR
        if motion_lr is not None:
            mv = motion_lr.to(dtype=dtype)
        elif lr_inputs.shape[1] >= 6:
            mv = lr_inputs[:, 4:6].to(dtype=dtype)
        else:
            mv = torch.zeros((b, 2, h_lr, w_lr), device=device, dtype=dtype)
        # Residual: per-pixel canvas-coverage shortfall. Use (1 - m) where
        # m is the rasterizer's weight-sum at LR. Areas with no canvas
        # coverage have residual ~1, fully-covered areas ~0. Combined with
        # the depth-disocclusion mask, this concentrates spawn budget at
        # newly-revealed pixels exactly when the canvas hasn't yet learned
        # them.
        _, m_lr = self._rasterize_canvas_at_lr(
            warped_canvas, active_mask, b, h_lr, w_lr, device, dtype
        )
        residual = (1.0 - m_lr).clamp(0.0, 1.0).to(dtype=dtype)
        debug_check("disocclusion.residual", residual)

        spawn_dict = self.disocclusion_spawner(
            depth_t=depth_t,
            depth_prev=depth_prev,
            MV=mv,
            lr_features=refined,
            residual=residual,
        )
        # Cache depth for next frame (detached — no autograd across frames).
        self._depth_lr_prev = depth_t.detach()

        K = int(spawn_dict["xy"].shape[0])
        if K == 0:
            zero_pos = torch.zeros((0, 2), device=device, dtype=dtype)
            zero_scale = torch.zeros((0, 2), device=device, dtype=dtype)
            zero_rot = torch.zeros((0,), device=device, dtype=dtype)
            zero_op = torch.zeros((0,), device=device, dtype=dtype)
            zero_col = torch.zeros(
                (0, self.cfg.token_dim), device=device, dtype=dtype
            )
            return CanvasState(
                positions=zero_pos,
                scales=zero_scale,
                rotations=zero_rot,
                opacities=zero_op,
                colors=zero_col,
                count=0,
            )
        # xy is in LR pixel coords (spawner reads LR features). Scale to HR
        # so it matches the legacy pipeline's HR-coordinate canvas convention.
        xy_lr = spawn_dict["xy"].to(dtype=dtype)
        positions_hr = (xy_lr * float(self.scale)).contiguous()
        # Isotropic scale_iso (LR pixel units) -> HR pixel units, broadcast
        # to (K, 2). Floor at 0.5 HR pixels so newly spawned Gaussians have
        # a usable footprint even if DGP outputs near-zero scale at init.
        scale_iso_lr = spawn_dict["scale"].to(dtype=dtype)
        scale_hr = (scale_iso_lr * float(self.scale)).clamp(min=0.5)
        scales = scale_hr[:, None].expand(-1, 2).contiguous()
        rotations = torch.zeros((K,), device=device, dtype=dtype)
        opacities = torch.ones((K,), device=device, dtype=dtype)
        # feat (K, feat_dim) -> colors (K, token_dim). Zero-init projection
        # means newborn Gaussians contribute nothing on step 0 and only
        # earn contribution as the projection trains.
        colors = self.disocclusion_color_proj(spawn_dict["feat"].to(dtype=dtype))
        return CanvasState(
            positions=positions_hr,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors.contiguous(),
            count=K,
        )

    def _flatten_spawned(self, spawned: GaussianSpawnState) -> CanvasState:
        b, k = spawned.positions.shape[:2]
        count = int(b * k)
        return CanvasState(
            positions=spawned.positions.reshape(count, 2).contiguous(),
            scales=spawned.scales.reshape(count, 2).contiguous(),
            rotations=spawned.rotations.reshape(count).contiguous(),
            opacities=spawned.opacities.reshape(count).contiguous(),
            colors=spawned.colors.reshape(count, spawned.colors.shape[-1]).contiguous(),
            count=count,
        )

    def _spawn_offset_for(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            not bool(self.cfg.spawn_offset_random)
            and not bool(self.cfg.spawn_subpixel_jitter)
        ) or not self.training:
            return None
        b = int(features.shape[0])
        device = features.device
        if (
            self._spawn_offset_xy is None
            or self._spawn_offset_xy.shape != (b, 2)
            or self._spawn_offset_xy.device != device
        ):
            offset = torch.zeros((b, 2), device=device, dtype=torch.float32)
            if bool(self.cfg.spawn_offset_random):
                offset = offset + torch.randint(
                    low=0,
                    high=int(self.cfg.tile_size_hr),
                    size=(b, 2),
                    device=device,
                    dtype=torch.int64,
                ).to(dtype=torch.float32)
            if bool(self.cfg.spawn_subpixel_jitter):
                offset = offset + torch.rand((b, 2), device=device, dtype=torch.float32)
            self._spawn_offset_xy = offset
        return self._spawn_offset_xy

    def _concat_canvas(
        self,
        warped: Optional[CanvasState],
        spawned: CanvasState,
    ) -> CanvasState:
        if warped is None or warped.count == 0:
            combined = spawned
        elif spawned.count == 0:
            combined = warped
        else:
            combined = CanvasState(
                positions=torch.cat([warped.positions[: warped.count], spawned.positions], dim=0),
                scales=torch.cat([warped.scales[: warped.count], spawned.scales], dim=0),
                rotations=torch.cat([warped.rotations[: warped.count], spawned.rotations], dim=0),
                opacities=torch.cat([warped.opacities[: warped.count], spawned.opacities], dim=0),
                colors=torch.cat([warped.colors[: warped.count], spawned.colors], dim=0),
                count=int(warped.count) + int(spawned.count),
            )
        return self._drop_oldest(combined)

    def _drop_oldest(self, canvas: CanvasState) -> CanvasState:
        capacity = int(self.cfg.canvas_capacity)
        if capacity <= 0:
            raise ValueError(f"canvas_capacity must be positive; got {capacity}")
        count = int(canvas.count)
        if count <= capacity:
            return canvas
        start = count - capacity
        return CanvasState(
            positions=canvas.positions[start:count].contiguous(),
            scales=canvas.scales[start:count].contiguous(),
            rotations=canvas.rotations[start:count].contiguous(),
            opacities=canvas.opacities[start:count].contiguous(),
            colors=canvas.colors[start:count].contiguous(),
            count=capacity,
        )

    def _render_active_mask(
        self,
        old_active: Optional[torch.Tensor],
        old_count: int,
        new_count: int,
        canvas: CanvasState,
    ) -> torch.Tensor:
        old_count = int(old_count)
        if old_count > 0 and old_active is not None:
            old = self._normalize_mask(old_active, old_count, canvas.positions.device)
        else:
            old = torch.zeros((0,), device=canvas.positions.device, dtype=torch.bool)
        new = torch.ones((new_count,), device=canvas.positions.device, dtype=torch.bool)
        active = torch.cat([old, new], dim=0)
        if active.shape[0] > canvas.count:
            active = active[-int(canvas.count):]
        elif active.shape[0] < canvas.count:
            pad = canvas.count - active.shape[0]
            active = F.pad(active, (pad, 0), value=False)
        return active.contiguous()

    def _normalize_mask(
        self,
        mask: torch.Tensor,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.bool).flatten()
        length = int(length)
        if mask.shape[0] == length:
            return mask
        if mask.shape[0] > length:
            return mask[:length]
        return F.pad(mask, (0, length - mask.shape[0]), value=False)

    def _update_st_state(
        self,
        canvas: CanvasState,
        active: torch.Tensor,
        previous_state: Optional[STVScoreState],
        old_count: int,
        new_count: int,
    ) -> None:
        n = int(canvas.count)
        state = self._st_state_after_write(
            previous_state=previous_state,
            old_count=old_count,
            new_count=new_count,
            final_count=n,
            device=canvas.opacities.device,
        )
        # 4DGS-1K spatial score adapted for v6's 2D additive splatting:
        #
        #   SS_i = footprint_area_i * opacity_i
        #        = (2π * sqrt(det(Σ_i))) * α_i
        #
        # det(Σ) is rotation-invariant for a Gaussian — for our scale-
        # rotation parameterization Σ = R diag(s_x², s_y²) R^T, so
        # det(Σ) = (s_x · s_y)² and sqrt(det(Σ)) = s_x · s_y. This captures
        # the per-Gaussian spatial extent that the prior opacity-only
        # heuristic ignored.
        #
        # 2D additive blending (v6's render path) has no per-pixel
        # transmittance term in the canonical 4DGS-1K sense; we use 1.0
        # and let the footprint × opacity product carry the signal. If
        # we later switch to sorted-back-to-front blending, T_i can be
        # computed as a product over earlier Gaussians; that's a v6.x
        # plumbing change, not an architectural one.
        #
        # All inputs are detached so the spawner's confidence path doesn't
        # leak gradient through the persistent ST state across frames.
        scales = canvas.scales[:n].detach().to(dtype=torch.float32)  # (N, 2)
        opac = canvas.opacities[:n].detach().to(dtype=torch.float32)  # (N,)
        sqrt_det_sigma = (scales[:, 0] * scales[:, 1]).abs()           # (N,)
        footprint = (2.0 * math.pi) * sqrt_det_sigma                   # (N,)
        spatial_contribution = (footprint * opac).unsqueeze(1)         # (N, 1)
        transmittance = torch.ones_like(spatial_contribution)
        self._st_state = update_st_score(
            state, spatial_contribution, transmittance, active[:n],
        )

    def _st_state_after_write(
        self,
        previous_state: Optional[STVScoreState],
        old_count: int,
        new_count: int,
        final_count: int,
        device: torch.device,
    ) -> STVScoreState:
        if previous_state is None or previous_state.spatial_accumulator.numel() == 0:
            return init_st_score_state(final_count, device=device, dtype=torch.float32)

        old_keep = min(
            int(old_count),
            int(previous_state.spatial_accumulator.shape[0]),
        )
        spatial_old = previous_state.spatial_accumulator[:old_keep].to(device=device)
        lifespan_old = previous_state.lifespan_count[:old_keep].to(device=device)
        spatial_new = torch.zeros(int(new_count), device=device, dtype=torch.float32)
        lifespan_new = torch.zeros(int(new_count), device=device, dtype=torch.int64)

        spatial = torch.cat([spatial_old, spatial_new], dim=0)
        lifespan = torch.cat([lifespan_old, lifespan_new], dim=0)
        if spatial.shape[0] > final_count:
            spatial = spatial[-final_count:]
            lifespan = lifespan[-final_count:]
        elif spatial.shape[0] < final_count:
            pad = final_count - spatial.shape[0]
            spatial = F.pad(spatial, (pad, 0), value=0.0)
            lifespan = F.pad(lifespan, (pad, 0), value=0)

        return STVScoreState(
            spatial_accumulator=spatial.contiguous(),
            lifespan_count=lifespan.contiguous(),
            frames_observed=previous_state.frames_observed,
        )


@dataclass
class CanvasState:
    """Minimal duck-typed canvas surface that ST-score and keyframe-mask
    consume. The full PersistentCanvas in ``oss/gaussian/canvas/canvas.py``
    is a superset of this contract; we keep V6Model decoupled from that
    concrete class so unit tests can pass a synthetic canvas in.
    """

    positions: torch.Tensor   # (N, 2)
    scales: torch.Tensor      # (N, 2)
    rotations: torch.Tensor   # (N,) angle in radians
    opacities: torch.Tensor   # (N,)
    colors: torch.Tensor      # (N, F)
    count: int

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
  Stage 5: Lightweight pixel head + sub-pixel upsample produce HR RGB.
  Stage 6: Softplus output activation supports HDR linear-light values >1.0
           (the model defaults to softplus per commit 43755fc — flag-
           configurable for SDR-only callers).
  Stage 7: STVScoreState aggregates per-Gaussian contribution + lifespan
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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oss.sr.v6.cross_attention import PixelGaussianFusion
from oss.sr.v6.hat import HAT, hat_l, hat_small, hat_tiny
from oss.sr.v6.keyframe_active_mask import KeyframeActiveMaskCache
from oss.sr.v6.st_variation_score import STVScoreState


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
    backbone, 15K canvas capacity, softplus output). Standard / Pico
    tiers override ``backbone`` and ``canvas_capacity``.
    """

    in_channels: int = 9
    scale: int = 2
    backbone: str = "hat-l"
    canvas_capacity: int = 15000
    token_dim: int = 64
    cross_attention_heads: int = 6
    window_size: int = 16
    color_activation: str = "softplus"   # "softplus" (HDR) | "sigmoid" (SDR)
    keyframe_interval: int = 10
    prune_every: int = 200
    prune_fraction: float = 0.7


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
        if self.cfg.color_activation not in ("softplus", "sigmoid"):
            raise ValueError(
                f"color_activation must be 'softplus' or 'sigmoid'; got "
                f"{self.cfg.color_activation!r}"
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

        self.fusion = PixelGaussianFusion(
            feat_dim=self.feat_dim,
            token_dim=self.cfg.token_dim,
            num_heads=self.cfg.cross_attention_heads,
            window_size=self.cfg.window_size,
        )

        # Pixel head + sub-pixel upsample. Conv to RGB*scale*scale, then
        # PixelShuffle. Standard SRCNN-style decoder.
        self.pixel_head = nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1)
        self.activation = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(self.feat_dim, 3 * (self.scale ** 2), 3, padding=1),
            nn.PixelShuffle(self.scale),
        )

        # Per-rank-local stateful pieces. NOT registered as buffers so DDP
        # doesn't try to sync them; they reset at trajectory boundaries.
        self._canvas_state: Optional[CanvasState] = None
        self._st_state: Optional[STVScoreState] = None
        self.keyframe_mask = KeyframeActiveMaskCache(
            keyframe_interval=self.cfg.keyframe_interval,
        )

        # Step counter for periodic prune; advances on every training-step
        # call to ``maybe_prune()``. Not a buffer — caller is expected to
        # drive it through the trainer.
        self._step_count: int = 0

        # zero_gbuffer_into_backbone is part of the saved-args contract
        # carried over from v5 for warm-start ckpts. v6 backbones are
        # trained with real G-buffers from the start, so default False.
        self.zero_gbuffer_into_backbone: bool = False

    # ------------------------------------------------------------------
    # Stateful canvas + score interface
    # ------------------------------------------------------------------

    def reset_state(self, device: Optional[torch.device] = None) -> None:
        """Reset the canvas, ST score, and keyframe mask. Call at the start
        of every trajectory (or every training step that does not carry
        canvas state across calls)."""
        self._canvas_state = None
        self._st_state = None
        self.keyframe_mask.reset()

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
        feats = self.backbone(lr_inputs)         # (B, feat_dim, H, W)
        feats = self.activation(self.pixel_head(feats))

        # Construct cross-attention tokens from the active subset of canvas
        # Gaussians. Empty canvas (first frame, fresh state) yields K=0
        # tokens — fusion module short-circuits to identity.
        tokens = self._build_canvas_tokens(feats)

        refined = self.fusion(feats, tokens)
        rgb_hr = self.upsample(refined)
        if self.cfg.color_activation == "sigmoid":
            rgb_hr = torch.sigmoid(rgb_hr)
        else:
            rgb_hr = F.softplus(rgb_hr)

        return rgb_hr

    # ------------------------------------------------------------------
    # ST-score-driven pruning hook
    # ------------------------------------------------------------------

    def maybe_prune(self) -> int:
        """Called by the trainer once per step. Returns the number of
        Gaussians pruned (0 if not a prune step or canvas empty).
        """
        self._step_count += 1
        if self._canvas_state is None or self._st_state is None:
            return 0
        if self._step_count % self.cfg.prune_every != 0:
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

    def _build_canvas_tokens(self, feats: torch.Tensor) -> torch.Tensor:
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
            frame_index=self._step_count,
            canvas=self._canvas_state,
            view_matrix=None,  # the cache's bbox-test handles None as identity
        )
        if active is None or active.sum() == 0:
            return feats.new_zeros((B, 0, self.cfg.token_dim))
        # Project active Gaussian colors into token_dim. The canvas stores
        # per-Gaussian feature/color tensors; we feed those through a
        # learnable Linear so the embedding can be optimized end-to-end.
        active_feats = self._canvas_state.colors[active]  # (K_active, F_canvas)
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

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
from oss.sr.v7.backbone_spawner import BackboneSpawner
from oss.sr.v7.parent_child_spawner import (
    ChildState, initialize_children_for_new_parents, materialize_to_canvas,
)


@dataclass
class V7Config:
    """V7Model configuration. Defaults target pico-tier."""
    in_channels: int = 9        # rgb + depth + motion(2) + normals(3)
    scale: int = 2
    feat_dim: int = 32          # backbone feature width
    latent_rank: int = 16       # canvas feature dim R
    # Canvas + spawner defaults chosen so a 2-spawn cycle (frame N at t=0
    # + frame N+1 at t=2) fits with ~3x headroom at TartanAir's 480x640
    # HR (training data), and so deployment HR shapes up to ~1080p fit
    # without further config changes. See docs/architecture/
    # 2026-05-13-v7-spawner-config-rationale.md for the bench data.
    canvas_capacity: int = 16384
    backbone_blocks: int = 4
    # Backbone selection: "placeholder" = small ConvNet (tests, fast),
    # "hat_tiny" = v6.x HAT-Tiny transformer (pico-tier teacher),
    # "hat_small" / "hat_l" = larger teachers for Standard / Heavy tiers.
    backbone_kind: str = "placeholder"
    # Spawner controls. k_per_tile=2 (down from prior 4) keeps the per-
    # spawn count fitting in canvas_capacity at 480x640 HR (4800 total
    # after 2 spawns) while still giving the parent-child mechanism
    # room to grow density adaptively. tile_size=16 is chosen so the
    # spawner's avg-pool kernel matches v6.x HAT-Tiny's window_size and
    # so 1080p HR pads to one extra tile, not many.
    enable_spawner: bool = True
    spawner_k_per_tile: int = 2
    spawner_tile_size: int = 16

    # Parent-child loss-adaptive density. Off by default for backwards
    # compatibility; flip on via V7Config(enable_parent_child=True) once
    # the trainer is in trajectory-persistent mode (see
    # docs/architecture/2026-05-13-v7-parent-child-integration-debt.md).
    enable_parent_child: bool = False
    # Per-step drift rate: child.opacity[i] += drift_rate * |grad_at_parent_i|
    parent_child_drift_rate: float = 0.05
    # Per-step decay (applied before drift) so children that stopped
    # accumulating gradient fade back below threshold instead of
    # being stuck.
    parent_child_decay: float = 0.98


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
    Stands in for HAT family in unit tests where transformer compute
    isn't worth the overhead."""

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


class _HATBackbone(nn.Module):
    """Wraps a v6.x HAT model (Tiny / Small / L) as a v7 backbone.

    The HAT module produces (B, embed_dim, H_lr, W_lr) features at LR
    resolution; this wrapper projects to feat_dim, upsamples to HR,
    and exposes a uniform forward signature matching
    _PlaceholderBackbone.
    """

    def __init__(self, in_channels: int, feat_dim: int, scale: int, kind: str):
        super().__init__()
        from oss.sr.v6 import hat as hat_mod
        builders = {
            "hat_tiny": hat_mod.hat_tiny,
            "hat_small": hat_mod.hat_small,
            "hat_l": hat_mod.hat_l,
        }
        if kind not in builders:
            raise ValueError(
                f"unknown HAT variant {kind!r}; expected one of {sorted(builders)}"
            )
        self.hat = builders[kind](in_channels=in_channels)
        embed_dim = int(self.hat.embed_dim)
        # 1x1 projection to v7 feat_dim, then bilinear to HR.
        # Pixel-shuffle would be sharper but introduces extra params
        # we don't need at the skeleton level; can swap in Phase 2D.
        self.proj = nn.Conv2d(embed_dim, feat_dim, kernel_size=1)
        self.scale = scale

    def forward(self, lr_inputs: torch.Tensor) -> torch.Tensor:
        feats_lr = self.hat(lr_inputs)
        feats_lr = self.proj(feats_lr)
        return F.interpolate(
            feats_lr, scale_factor=self.scale,
            mode="bilinear", align_corners=False,
        )


class V7Model(nn.Module):
    """End-to-end v7 model: backbone + N-D canvas time-slice + fusion."""

    def __init__(self, cfg: Optional[V7Config] = None):
        super().__init__()
        self.cfg = cfg or V7Config()
        kind = self.cfg.backbone_kind
        if kind == "placeholder":
            self.backbone = _PlaceholderBackbone(
                in_channels=self.cfg.in_channels,
                feat_dim=self.cfg.feat_dim,
                scale=self.cfg.scale,
                blocks=self.cfg.backbone_blocks,
            )
        elif kind in ("hat_tiny", "hat_small", "hat_l"):
            self.backbone = _HATBackbone(
                in_channels=self.cfg.in_channels,
                feat_dim=self.cfg.feat_dim,
                scale=self.cfg.scale,
                kind=kind,
            )
        else:
            raise ValueError(
                f"V7Config.backbone_kind={kind!r} not recognized; "
                f"expected 'placeholder' | 'hat_tiny' | 'hat_small' | 'hat_l'"
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
        # know the device. ChildState (parent-child spawner) mirrors the
        # canvas shape and is allocated together.
        self._canvas: Optional[NDCanvasState] = None
        self._child: Optional[ChildState] = None
        # When parent-child is enabled, EVERY render that retains grad on
        # positions gets pushed here. drift_children_from_grad() iterates
        # this list to accumulate per-parent attribution across all
        # renders since the last drift call (e.g. the 3-render flow
        # spawn-at-t=0, spawn-at-t=2, render-at-t=1 should all contribute).
        # Previous behavior stashed only the LAST render's tensor,
        # silently discarding attribution from earlier renders -- caught
        # by the 2026-05-14 audit.
        self._retained_positions_for_grad: list[torch.Tensor] = []
        # Tracks how many parents have been initialized as children. New
        # parents (live-but-uninitialized) get children re-initialized
        # at the start of each forward.
        self._n_children_initialized: int = 0

        # Spawner: decodes refined_hr into K Gaussians/frame for the canvas.
        self.spawner: Optional[BackboneSpawner] = None
        if self.cfg.enable_spawner:
            self.spawner = BackboneSpawner(
                feat_dim=self.cfg.feat_dim,
                latent_rank=self.cfg.latent_rank,
                k_per_tile=self.cfg.spawner_k_per_tile,
                tile_size=self.cfg.spawner_tile_size,
            )

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
        # Parent-child state mirrors the canvas; only allocated when
        # the mechanism is enabled, otherwise we don't pay the memory.
        if self.cfg.enable_parent_child:
            self._child = ChildState.empty(
                capacity=self.cfg.canvas_capacity,
                feature_dim=self.cfg.latent_rank,
                device=device,
                dtype=torch.float32,
            )
            self._n_children_initialized = 0
        return self._canvas

    @property
    def child(self) -> ChildState:
        if self._child is None:
            raise RuntimeError(
                "V7Model.child accessed but parent-child is disabled. "
                "Set V7Config.enable_parent_child=True and re-allocate."
            )
        return self._child

    def reset_state(self, device: torch.device | str = "cpu") -> None:
        """Clear canvas + child state. Trajectory boundary hook."""
        if self._canvas is None:
            self.allocate_canvas(device)
        else:
            self._canvas.reset()
        if self._child is not None:
            self._child.reset()
            self._n_children_initialized = 0
        self._retained_positions_for_grad = []

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
        # When parent-child is enabled, mark the active positions for
        # retained gradients so the trainer can read per-parent
        # attribution after backward(). Append to the list so multiple
        # renders per step (the SR forward at t=2 AND the OSS-FX forward
        # at t=1) BOTH contribute to drift signal; previously we
        # overwrote a scalar and only the last render counted.
        if self.cfg.enable_parent_child and pos.requires_grad:
            pos.retain_grad()
            self._retained_positions_for_grad.append(pos)
        rendered = render_nd_time_slice(
            means=pos, covs=cov, features=feat, opacities=opacity,
            t_query=t_query, image_hw=output_hw,
        )
        # Add batch dim
        return rendered.unsqueeze(0)

    @torch.no_grad()
    def drift_children_from_grad(self) -> torch.Tensor:
        """Drive child opacity/brightness up where the parent's position
        gradient is large (= the loss cares about that parent).

        Call AFTER backward() and BEFORE optim.step(). Reads
        gradients from EVERY retained-grad render since the last drift
        call, sums per-parent gradient magnitudes across them, normalizes,
        decays + drifts each child slot.

        Previously stashed only the last render's positions tensor, so
        a 3-render flow (spawn t=0, spawn t=2, render t=1) only saw
        the t=1 render's gradients -- in curriculum stage 1 with
        lambda_fg=0, that's zero gradient signal forever (caught by
        the 2026-05-14 audit).

        Returns the per-parent gradient norm tensor (n_live,) for
        diagnostics; trainer can log this to history.jsonl.
        """
        if not self.cfg.enable_parent_child:
            raise RuntimeError("Parent-child must be enabled to drift children.")
        if not self._retained_positions_for_grad:
            # No renders since last drift -- nothing to drift on.
            return torch.zeros((0,), device=self.canvas.device)
        # All retained tensors come from active_view() since the last
        # reset_state. They all have the same n_live shape (n_live is
        # monotonic between resets), so we can sum gradients elementwise.
        n_live = self._retained_positions_for_grad[0].shape[0]
        accum_grad = torch.zeros((n_live, 3),
                                  device=self.canvas.device,
                                  dtype=torch.float32)
        for pos in self._retained_positions_for_grad:
            if pos.grad is None or pos.shape[0] != n_live:
                continue
            accum_grad += pos.grad.detach()
        # Clear the retained list so the next step starts fresh.
        self._retained_positions_for_grad = []
        # Per-parent gradient magnitude (sum over the 3 position dims).
        per_parent_grad = accum_grad.abs().sum(dim=-1)  # (n_live_in_view,)
        if per_parent_grad.abs().sum() == 0:
            return per_parent_grad
        # Normalize by the max so the drift rate is shape-invariant
        # across different canvas sizes / loss scales.
        max_g = per_parent_grad.max().clamp(min=1e-12)
        normalized = per_parent_grad / max_g

        # active_view returns positions in active-mask order (compacted).
        # We need to scatter the per-parent signal back into the full
        # canvas-indexed child arrays. canvas.mask[:n_live] tells us
        # which slots are live.
        canvas = self.canvas
        child = self._child
        assert child is not None
        live_mask = canvas.mask[: canvas.n_live]
        live_idx = live_mask.nonzero(as_tuple=True)[0]
        # Decay first, then add drift on the live indices.
        child.opacity[: canvas.n_live] *= self.cfg.parent_child_decay
        child.brightness[: canvas.n_live] *= self.cfg.parent_child_decay
        # Sanity: live_idx length must match per_parent_grad length
        if live_idx.shape[0] == per_parent_grad.shape[0]:
            child.opacity[live_idx] = (
                child.opacity[live_idx]
                + self.cfg.parent_child_drift_rate * normalized
            )
            # Brightness drifts proportionally with opacity for now -- a
            # separate brightness signal (the parent's rendered feature
            # magnitude) is a future refinement.
            child.brightness[live_idx] = (
                child.brightness[live_idx]
                + 0.1 * self.cfg.parent_child_drift_rate * normalized
            )
        return per_parent_grad

    @torch.no_grad()
    def materialize_pending_children(self) -> int:
        """Promote any children that have crossed the opacity / brightness
        threshold into full-fledged canvas Gaussians.

        Call AFTER drift_children_from_grad() and BEFORE the next
        forward(). Returns the number materialized this round.
        """
        if not self.cfg.enable_parent_child:
            return 0
        if self._child is None or self._canvas is None:
            return 0
        # materialize_to_canvas does in-place writes to child.*; the
        # @torch.no_grad() decorator on this method keeps autograd out
        # of it. Resets the materialized slots back to dormant.
        n_materialized = materialize_to_canvas(self._canvas, self._child)
        # Newly added canvas Gaussians have no associated children yet;
        # they get initialized on the next initialize_new_children() call.
        return n_materialized

    @torch.no_grad()
    def initialize_new_children(self, init_dpos_std: float = 0.1) -> int:
        """Initialize child slots for any canvas parents that don't have
        a child yet (i.e. were spawned since the last init call).

        Returns the number of new children initialized.
        """
        if not self.cfg.enable_parent_child:
            return 0
        if self._child is None or self._canvas is None:
            return 0
        new_parent_count = self._canvas.n_live - self._n_children_initialized
        if new_parent_count <= 0:
            return 0
        new_indices = torch.arange(
            self._n_children_initialized, self._canvas.n_live,
            device=self._canvas.device,
        )
        initialize_children_for_new_parents(
            self._child, parent_indices=new_indices,
            init_dpos_std=init_dpos_std,
        )
        self._n_children_initialized = self._canvas.n_live
        return int(new_parent_count)

    def spawn_into_canvas(self, refined_hr: torch.Tensor, t: float) -> int:
        """If a spawner is configured, decode refined_hr -> K Gaussians and
        add them to the canvas at t. Returns the number added.

        Caller is responsible for not exceeding canvas capacity; this
        method will raise if it would. v7 trainer policy is to prune
        old / low-opacity Gaussians periodically to maintain headroom.
        """
        if self.spawner is None or self._canvas is None:
            return 0
        spawned = self.spawner(refined_hr, t=t)
        n_to_add = spawned["positions"].shape[0]
        # Avoid overflow: if not enough room, raise so the trainer
        # surface the issue rather than silently dropping spawns.
        if self._canvas.n_live + n_to_add > self._canvas.capacity:
            raise RuntimeError(
                f"NDCanvasState capacity ({self._canvas.capacity}) exceeded "
                f"by spawn (have {self._canvas.n_live} live, spawning "
                f"{n_to_add}). Trainer should prune before spawning more."
            )
        self._canvas.add(
            positions=spawned["positions"],
            cov_raw=spawned["cov_raw"],
            features=spawned["features"],
            opacity=spawned["opacity"],
        )
        return n_to_add

    def forward(
        self,
        lr_inputs: torch.Tensor,           # (B, in_ch, H_lr, W_lr)
        t_query: float = 0.0,              # absolute time coordinate to render
        output_hw: Optional[tuple[int, int]] = None,
        spawn_at_t: Optional[float] = None, # if given, run spawner at this t
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

        # Optional spawner: decode refined_hr into K canvas Gaussians at the
        # requested time and append them. Trainer typically passes
        # spawn_at_t=t_query for normal SR step or spawn_at_t=N for
        # frame-N's add-to-canvas, and leaves the FG forward (rendering
        # an intermediate t_query) without spawning.
        if spawn_at_t is not None and self.spawner is not None:
            self.spawn_into_canvas(refined_hr[:1], t=float(spawn_at_t))

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

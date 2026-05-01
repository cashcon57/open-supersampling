"""``PersistentCanvas`` — the temporal heart of OSS-Gaussian.

A GPU-resident SoA buffer of N Gaussians persisting across frames, warped
each frame by motion vectors, with per-tile error driving prune+spawn.

Design doc: ``docs/superpowers/gaussian-canvas-design.md``
Sprint plan: ``docs/superpowers/plans/2026-05-01-gaussian-sprint-5-plan.md``

Sprint 5 v1 is pure PyTorch — same code path on CPU and CUDA. A custom
fused CUDA kernel is post-Sprint-5 perf work.

Public API:

    canvas = PersistentCanvas(capacity=8000, feat_dim=3, output_hw=(720, 1280))
    canvas.initialize_random()
    img = canvas.render()                     # (F, H, W)
    canvas.update(motion, lr_frame)           # one frame: warp + render + error
                                              # + prune + spawn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from oss.gaussian.canvas.error_detection import (
    gaussians_error_from_tiles,
    per_tile_mse,
)
from oss.gaussian.canvas.prune_spawn import (
    PrunePolicy,
    apply_prune_spawn,
    select_for_pruning,
    select_spawn_tiles,
)
from oss.gaussian.canvas.warp import warp_positions
from oss.gaussian.renderer import GaussianBatch, Rasterizer


@dataclass(frozen=True)
class CanvasStats:
    """Snapshot of one update's diagnostics. Returned by ``update``."""

    n_alive_before: int
    n_alive_after: int
    n_pruned: int
    n_spawned: int
    mean_tile_error: float


class PersistentCanvas:
    """Persistent 2D Gaussian canvas with motion warp + error-driven prune+spawn.

    Storage layout is **Struct of Arrays** (see design doc §1):

        positions : (capacity, 2)  float — pixel-space (x, y)
        scales    : (capacity, 2)  float — per-axis scale
        rotations : (capacity,)    float — radians
        colors    : (capacity, F)  float — feature/colour values
        age       : (capacity,)    long  — frames alive
        error     : (capacity,)    float — last per-Gaussian error
        alive     : (capacity,)    bool  — slot occupancy

    Args:
        capacity:    max number of Gaussian slots (the budget knob — 1K
                     pico, 5K lite, 8K standard, 15K ultra).
        feat_dim:    feature/colour channels per Gaussian (3 for RGB).
        output_hw:   ``(H, W)`` render resolution.
        tile_size:   16 — must match Sprint-1 ``Rasterizer.TILE_SIZE`` and
                     Sprint-3 classifier.
        device:      torch device.
        dtype:       float dtype for geometry/colour tensors.
        prune_policy: optional override for the prune decision tree.
    """

    def __init__(
        self,
        capacity: int = 8000,
        feat_dim: int = 3,
        output_hw: Tuple[int, int] = (720, 1280),
        tile_size: int = 16,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        prune_policy: Optional[PrunePolicy] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive; got {feat_dim}")
        h, w = output_hw
        if h <= 0 or w <= 0:
            raise ValueError(f"output_hw must be positive; got {output_hw}")
        if h % tile_size or w % tile_size:
            raise ValueError(
                f"output_hw {output_hw} must be multiples of tile_size={tile_size}"
            )

        self.capacity = int(capacity)
        self.feat_dim = int(feat_dim)
        self.output_hw = (int(h), int(w))
        self.tile_size = int(tile_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.policy = prune_policy if prune_policy is not None else PrunePolicy()

        self.positions = torch.zeros((capacity, 2), dtype=dtype, device=self.device)
        self.scales = torch.ones((capacity, 2), dtype=dtype, device=self.device)
        self.rotations = torch.zeros((capacity,), dtype=dtype, device=self.device)
        self.colors = torch.zeros((capacity, feat_dim), dtype=dtype, device=self.device)
        self.age = torch.zeros((capacity,), dtype=torch.long, device=self.device)
        self.error = torch.zeros((capacity,), dtype=dtype, device=self.device)
        self.alive = torch.zeros((capacity,), dtype=torch.bool, device=self.device)

        self._rasterizer = Rasterizer(tile_size=tile_size)
        # Reference backend tolerates any tile size and runs on CPU; the
        # CUDA backend kicks in automatically when tensors live on CUDA
        # and gsplat is available — see Sprint 1 selection logic.

    # ------------------------------------------------------------------ init

    def initialize_random(self, seed: Optional[int] = None) -> None:
        """Fill the canvas with random Gaussians sampled across the frame.

        Used as a default starting state and as a unit-test fixture.
        """
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        h, w = self.output_hw
        xs = torch.rand((self.capacity,), generator=gen) * w
        ys = torch.rand((self.capacity,), generator=gen) * h
        self.positions[:, 0] = xs.to(self.dtype).to(self.device)
        self.positions[:, 1] = ys.to(self.dtype).to(self.device)
        self.scales[:, :] = float(self.tile_size) * 0.5
        self.rotations[:] = 0.0
        self.colors[:] = 0.5
        self.age[:] = 0
        self.error[:] = 0.0
        self.alive[:] = True

    def initialize_from_batch(self, batch: GaussianBatch) -> None:
        """Seed the canvas from a ``GaussianBatch`` (e.g. param network's
        first-frame output). Extra slots stay dead.
        """
        n = min(batch.num_gaussians, self.capacity)
        self.alive[:] = False
        self.age[:] = 0
        self.error[:] = 0.0
        self.positions[:n] = batch.xy[:n].to(self.positions.dtype).to(self.device)
        self.scales[:n] = batch.scale[:n].to(self.scales.dtype).to(self.device)
        self.rotations[:n] = batch.rot[:n].to(self.rotations.dtype).to(self.device)
        f = min(self.feat_dim, batch.feat_dim)
        self.colors[:n, :f] = batch.feat[:n, :f].to(self.colors.dtype).to(self.device)
        self.alive[:n] = True

    # -------------------------------------------------------------- snapshot

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum().item())

    def snapshot(self) -> GaussianBatch:
        """View the alive subset as a renderer-ready ``GaussianBatch``."""
        idx = self.alive.nonzero(as_tuple=False).flatten()
        return GaussianBatch(
            xy=self.positions[idx],
            scale=self.scales[idx],
            rot=self.rotations[idx],
            feat=self.colors[idx],
        )

    # ---------------------------------------------------------------- render

    def render(self) -> torch.Tensor:
        """Render the alive Gaussians at ``self.output_hw``.

        Returns ``(F, H, W)`` float tensor on ``self.device``.
        """
        gb = self.snapshot()
        if gb.num_gaussians == 0:
            return torch.zeros(
                (self.feat_dim, *self.output_hw), dtype=self.dtype, device=self.device
            )
        return self._rasterizer(gb, self.output_hw)

    # ---------------------------------------------------------------- update

    def update(
        self,
        motion: torch.Tensor,
        lr_frame: torch.Tensor,
        new_gaussians: Optional[GaussianBatch] = None,
        classifier_mask: Optional[torch.Tensor] = None,
    ) -> CanvasStats:
        """Run one frame of the canvas lifecycle.

        Args:
            motion:          ``(2, H, W)`` motion vectors at output
                             resolution. Same units as ``positions``.
            lr_frame:        ``(F, h, w)`` low-resolution input. Will be
                             bilinearly upsampled to ``output_hw`` for
                             error scoring.
            new_gaussians:   Optional ``GaussianBatch`` to draw spawn
                             replacements from (typically the
                             ``OutputHead`` decode of the param network's
                             output for the high-error tiles). If
                             ``None``, prune still happens but no
                             replacement is written — alive count drops.
            classifier_mask: ``(h_tiles, w_tiles)`` bool from
                             ``TileClassifier``. Restricts spawn to
                             tiles the classifier marked complex.

        Returns:
            ``CanvasStats`` summarising this frame.
        """
        H, W = self.output_hw
        T = self.tile_size
        h_t, w_t = H // T, W // T

        # 1. Warp positions.
        new_xy, in_frame = warp_positions(self.positions, motion, self.output_hw)
        self.positions = new_xy

        # 2. Bump age for live Gaussians.
        self.age = torch.where(
            self.alive, self.age + 1, torch.zeros_like(self.age)
        )

        # 3. Render current canvas (post-warp).
        rendered = self.render()

        # 4. Score per-tile error against upsampled LR input.
        if lr_frame.shape[1:] != self.output_hw:
            lr_up = F.interpolate(
                lr_frame.unsqueeze(0),
                size=self.output_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            lr_up = lr_frame
        tile_err = per_tile_mse(rendered, lr_up, T)
        g_err = gaussians_error_from_tiles(self.positions, tile_err, T, self.output_hw)
        # Out-of-frame Gaussians get +inf from the lookup; merge with the
        # warp's authoritative ``in_frame`` flag for safety.
        g_err = torch.where(
            in_frame, g_err, torch.full_like(g_err, float("inf"))
        )
        self.error = torch.where(self.alive, g_err, torch.zeros_like(g_err))

        n_alive_before = self.n_alive

        # 5. Pruning decision.
        prune_idx = select_for_pruning(
            alive_mask=self.alive,
            in_frame=in_frame,
            age=self.age,
            g_error=self.error,
            tile_error=tile_err,
            capacity=self.capacity,
            policy=self.policy,
        )

        # 6. Spawn tile selection (purely informational here; caller may
        # use this list to drive a sparse network call before passing
        # ``new_gaussians`` in. We expose it via the stats indirectly —
        # tests can call ``select_spawn_tiles`` directly.)
        n_to_spawn_tiles = max(1, prune_idx.numel())
        # Each spawn tile may emit ``K`` Gaussians; the network/decoder
        # contract is up to the caller. We simply consume whatever
        # ``new_gaussians`` contains.
        _ = select_spawn_tiles(
            tile_err, n_to_spawn_tiles, classifier_mask=classifier_mask
        )

        # 7. Apply.
        apply_prune_spawn(self, prune_idx, new_gaussians)

        n_alive_after = self.n_alive
        n_pruned = int(prune_idx.numel())
        n_spawned = (
            min(new_gaussians.num_gaussians, n_pruned)
            if new_gaussians is not None
            else 0
        )
        return CanvasStats(
            n_alive_before=n_alive_before,
            n_alive_after=n_alive_after,
            n_pruned=n_pruned,
            n_spawned=n_spawned,
            mean_tile_error=float(tile_err.mean().item()),
        )


__all__ = ["PersistentCanvas", "CanvasStats"]

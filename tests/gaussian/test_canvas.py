"""Sprint 5 persistent canvas tests.

All tests run on CPU. Synthetic inputs make expected outcomes
deterministic — no Sprint 2 captures required.

Spec: docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md
Plan: docs/superpowers/plans/2026-05-01-gaussian-sprint-5-plan.md
"""

from __future__ import annotations

import math

import pytest
import torch

from oss.gaussian.canvas import (
    CanvasStats,
    PersistentCanvas,
    PrunePolicy,
    apply_prune_spawn,
    gaussians_error_from_tiles,
    per_tile_mse,
    select_for_pruning,
    select_spawn_tiles,
    warp_canvas,
    warp_positions,
)
from oss.gaussian.renderer import GaussianBatch


# ----------------------------------------------------------------- helpers

H, W = 64, 64
T = 16


def _make_canvas(capacity: int = 32, output_hw=(H, W)) -> PersistentCanvas:
    canvas = PersistentCanvas(
        capacity=capacity,
        feat_dim=3,
        output_hw=output_hw,
        tile_size=T,
        device="cpu",
        dtype=torch.float32,
    )
    return canvas


def _zero_motion(h: int = H, w: int = W) -> torch.Tensor:
    return torch.zeros(2, h, w)


def _constant_motion(dx: float, dy: float, h: int = H, w: int = W) -> torch.Tensor:
    m = torch.zeros(2, h, w)
    m[0, :, :] = dx
    m[1, :, :] = dy
    return m


# ============================================================ canvas init


class TestCanvasInit:
    def test_random_init_alive_count(self) -> None:
        canvas = _make_canvas(capacity=64)
        canvas.initialize_random(seed=0)
        assert canvas.n_alive == 64

    def test_random_init_positions_inside_frame(self) -> None:
        canvas = _make_canvas(capacity=64)
        canvas.initialize_random(seed=1)
        xs = canvas.positions[:, 0]
        ys = canvas.positions[:, 1]
        assert (xs >= 0).all() and (xs < W).all()
        assert (ys >= 0).all() and (ys < H).all()

    def test_init_from_batch(self) -> None:
        n = 5
        batch = GaussianBatch(
            xy=torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0],
                             [40.0, 40.0], [50.0, 50.0]]),
            scale=torch.full((n, 2), 4.0),
            rot=torch.zeros(n),
            feat=torch.full((n, 3), 0.7),
        )
        canvas = _make_canvas(capacity=10)
        canvas.initialize_from_batch(batch)
        assert canvas.n_alive == n
        assert torch.allclose(canvas.positions[:n], batch.xy)
        assert torch.allclose(canvas.colors[:n], batch.feat)


# ============================================================ motion warp


class TestMotionWarp:
    def test_zero_motion_no_movement(self) -> None:
        xy = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        new_xy, in_frame = warp_positions(xy, _zero_motion(), (H, W))
        assert torch.allclose(new_xy, xy, atol=1e-5)
        assert in_frame.all()

    def test_constant_motion_uniform_shift(self) -> None:
        dx, dy = 3.5, -2.0
        xy = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        new_xy, in_frame = warp_positions(xy, _constant_motion(dx, dy), (H, W))
        expected = xy + torch.tensor([[dx, dy], [dx, dy]])
        assert torch.allclose(new_xy, expected, atol=1e-4)
        assert in_frame.all()

    def test_out_of_frame_flag(self) -> None:
        # Position 60, +10 shift → 70 > W=64 → out.
        xy = torch.tensor([[60.0, 30.0], [10.0, 10.0]])
        new_xy, in_frame = warp_positions(xy, _constant_motion(10.0, 0.0), (H, W))
        assert in_frame[0].item() is False
        assert in_frame[1].item() is True

    def test_warp_canvas_alpha_scaling(self) -> None:
        canvas = _make_canvas(capacity=4)
        canvas.initialize_from_batch(GaussianBatch(
            xy=torch.tensor([[10.0, 10.0], [20.0, 20.0],
                             [30.0, 30.0], [40.0, 40.0]]),
            scale=torch.full((4, 2), 4.0),
            rot=torch.zeros(4),
            feat=torch.full((4, 3), 0.5),
        ))
        motion = _constant_motion(4.0, 0.0)
        warped_full = warp_canvas(canvas, motion, alpha=1.0)
        warped_half = warp_canvas(canvas, motion, alpha=0.5)
        # Full alpha shifts by 4; half alpha shifts by 2.
        assert torch.allclose(
            warped_full.positions[:4, 0] - canvas.positions[:4, 0],
            torch.full((4,), 4.0),
            atol=1e-4,
        )
        assert torch.allclose(
            warped_half.positions[:4, 0] - canvas.positions[:4, 0],
            torch.full((4,), 2.0),
            atol=1e-4,
        )

    def test_warp_canvas_does_not_mutate_source(self) -> None:
        canvas = _make_canvas(capacity=4)
        canvas.initialize_random(seed=0)
        before = canvas.positions.clone()
        _ = warp_canvas(canvas, _constant_motion(5.0, 0.0), alpha=1.0)
        assert torch.allclose(canvas.positions, before)


# ============================================================ error detection


class TestErrorDetection:
    def test_blank_vs_random_high_error(self) -> None:
        rendered = torch.zeros(3, H, W)
        torch.manual_seed(0)
        target = torch.rand(3, H, W)
        err = per_tile_mse(rendered, target, T)
        assert err.shape == (H // T, W // T)
        assert (err > 0.0).all()
        assert err.mean() > 0.05

    def test_matched_low_error(self) -> None:
        torch.manual_seed(1)
        target = torch.rand(3, H, W)
        err = per_tile_mse(target, target, T)
        assert err.max().item() == pytest.approx(0.0, abs=1e-7)

    def test_per_gaussian_lookup(self) -> None:
        # Build tile_err with one bright tile.
        h_t, w_t = H // T, W // T
        tile_err = torch.zeros(h_t, w_t)
        tile_err[1, 2] = 9.0
        # A Gaussian sitting in tile (1, 2) should report 9.0.
        # Tile (1, 2) covers x ∈ [32, 48), y ∈ [16, 32).
        xy = torch.tensor([[40.0, 24.0], [4.0, 4.0]])
        g_err = gaussians_error_from_tiles(xy, tile_err, T, (H, W))
        assert g_err[0].item() == pytest.approx(9.0)
        assert g_err[1].item() == pytest.approx(0.0)

    def test_per_gaussian_out_of_frame_inf(self) -> None:
        tile_err = torch.zeros(H // T, W // T)
        xy = torch.tensor([[1000.0, 1000.0]])
        g_err = gaussians_error_from_tiles(xy, tile_err, T, (H, W))
        assert math.isinf(g_err[0].item())


# ============================================================ pruning


class TestPruning:
    def _baseline_state(self, n: int = 16):
        alive = torch.ones(n, dtype=torch.bool)
        in_frame = torch.ones(n, dtype=torch.bool)
        age = torch.full((n,), 5, dtype=torch.long)
        g_err = torch.full((n,), 0.1)
        tile_err = torch.zeros(2, 2)
        return alive, in_frame, age, g_err, tile_err

    def test_prune_out_of_frame(self) -> None:
        alive, in_frame, age, g_err, tile_err = self._baseline_state()
        in_frame[3] = False
        in_frame[7] = False
        idx = select_for_pruning(
            alive, in_frame, age, g_err, tile_err, capacity=16,
            policy=PrunePolicy(max_prune_per_frame_frac=1.0),
        )
        sel = set(idx.tolist())
        assert 3 in sel and 7 in sel

    def test_prune_aged_low_contrib(self) -> None:
        alive, in_frame, age, g_err, tile_err = self._baseline_state()
        # Bump one Gaussian above age_max with low error.
        age[5] = 100
        g_err[5] = 0.01  # very low
        # Make others have higher error so 5 lands below the 75th pct.
        g_err[0:5] = 0.5
        g_err[6:] = 0.5
        idx = select_for_pruning(
            alive, in_frame, age, g_err, tile_err, capacity=16,
            policy=PrunePolicy(age_max=60, max_prune_per_frame_frac=1.0),
        )
        assert 5 in idx.tolist()

    def test_prune_high_tile_error(self) -> None:
        alive, in_frame, age, g_err, tile_err = self._baseline_state()
        # One Gaussian in a high-error tile.
        g_err[2] = 10.0
        g_err[~(torch.arange(16) == 2)] = 0.05
        idx = select_for_pruning(
            alive, in_frame, age, g_err, tile_err, capacity=16,
            policy=PrunePolicy(max_prune_per_frame_frac=1.0),
        )
        assert 2 in idx.tolist()

    def test_prune_budget_clamp(self) -> None:
        alive, in_frame, age, g_err, tile_err = self._baseline_state(n=100)
        in_frame[:50] = False  # 50 candidates; budget caps at 5.
        idx = select_for_pruning(
            alive, in_frame, age, g_err, tile_err, capacity=100,
            policy=PrunePolicy(max_prune_per_frame_frac=0.05),
        )
        assert idx.numel() == 5

    def test_prune_skips_dead_slots(self) -> None:
        alive, in_frame, age, g_err, tile_err = self._baseline_state()
        alive[0] = False
        in_frame[0] = False  # dead + out of frame
        idx = select_for_pruning(
            alive, in_frame, age, g_err, tile_err, capacity=16,
            policy=PrunePolicy(max_prune_per_frame_frac=1.0),
        )
        assert 0 not in idx.tolist()


# ============================================================ spawning


class TestSpawning:
    def test_select_spawn_tiles_top_error(self) -> None:
        tile_err = torch.tensor([[0.1, 0.2, 9.0],
                                 [0.0, 5.0, 0.0]])
        coords = select_spawn_tiles(tile_err, n_tiles=2)
        # Expect (0, 2) and (1, 1).
        rows = {tuple(c.tolist()) for c in coords}
        assert (0, 2) in rows and (1, 1) in rows

    def test_select_spawn_tiles_classifier_mask(self) -> None:
        tile_err = torch.tensor([[0.0, 9.0],
                                 [0.0, 0.0]])
        # Mask says (0, 1) is simple → must not be picked.
        mask = torch.tensor([[True, False], [True, True]])
        coords = select_spawn_tiles(tile_err, n_tiles=2, classifier_mask=mask)
        assert (0, 1) not in {tuple(c.tolist()) for c in coords}

    def test_apply_prune_spawn_capacity_invariant(self) -> None:
        canvas = _make_canvas(capacity=8)
        canvas.initialize_random(seed=42)
        # Prune two slots, supply two replacement Gaussians.
        prune_idx = torch.tensor([0, 1], dtype=torch.long)
        new_g = GaussianBatch(
            xy=torch.tensor([[5.0, 5.0], [6.0, 6.0]]),
            scale=torch.full((2, 2), 4.0),
            rot=torch.zeros(2),
            feat=torch.full((2, 3), 0.9),
        )
        apply_prune_spawn(canvas, prune_idx, new_g)
        assert canvas.n_alive == 8  # one-to-one replacement
        # New Gaussians should now occupy slots 0 and 1.
        assert torch.allclose(
            canvas.positions[0], torch.tensor([5.0, 5.0]), atol=1e-5
        )

    def test_apply_prune_only_no_spawn(self) -> None:
        canvas = _make_canvas(capacity=4)
        canvas.initialize_random(seed=0)
        apply_prune_spawn(canvas, torch.tensor([0, 1], dtype=torch.long), None)
        assert canvas.n_alive == 2


# ============================================================ integration


class TestIntegration:
    def test_one_frame_lowers_error(self) -> None:
        """End-to-end: build a deliberately-wrong canvas, run one update
        round with a stub-network spawn that matches the LR, assert
        post-update error is meaningfully lower than pre-update."""
        torch.manual_seed(0)
        # Loosen the age floor so the single-frame integration test
        # exercises rule R3 — the multi-frame age dynamics are covered
        # separately by ``test_capacity_stable_across_frames``.
        canvas = _make_canvas(capacity=4)
        canvas.policy = PrunePolicy(min_age_before_prune=0,
                                    tile_error_pct=0.25,
                                    max_prune_per_frame_frac=1.0)

        # Wrong-state canvas: a single bright Gaussian sitting in a tile
        # where the LR is dark. Its tile error is therefore high and R3
        # fires. Other slots stay dead.
        bad = GaussianBatch(
            xy=torch.tensor([[8.0, 8.0]]),
            scale=torch.tensor([[4.0, 4.0]]),
            rot=torch.zeros(1),
            feat=torch.tensor([[1.0, 1.0, 1.0]]),
        )
        canvas.initialize_from_batch(bad)

        # LR frame with a bright square in the middle (around 32, 32).
        # The corner tile that holds the (8, 8) Gaussian is dark in LR
        # → high error, R3 fires.
        lr = torch.zeros(3, H, W)
        lr[:, 24:40, 24:40] = 1.0

        pre_err = per_tile_mse(canvas.render(), lr, T).mean().item()

        # "Network output" — one Gaussian in the middle that matches the
        # bright square.
        spawn = GaussianBatch(
            xy=torch.tensor([[32.0, 32.0]]),
            scale=torch.tensor([[6.0, 6.0]]),
            rot=torch.zeros(1),
            feat=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        stats = canvas.update(motion=_zero_motion(), lr_frame=lr,
                              new_gaussians=spawn)

        post_err = per_tile_mse(canvas.render(), lr, T).mean().item()
        assert isinstance(stats, CanvasStats)
        assert stats.n_alive_after >= stats.n_alive_before - stats.n_pruned
        # Sanity — at least some prunes happened given gross mismatch.
        assert stats.n_pruned >= 1
        # Error should drop meaningfully.
        assert post_err < pre_err * 0.8, (
            f"expected post_err < 0.8 * pre_err; pre={pre_err:.4f}, post={post_err:.4f}"
        )

    def test_capacity_stable_across_frames(self) -> None:
        """Repeatedly updating with neutral inputs should keep the alive
        count inside ``[capacity - max_prune, capacity]``."""
        torch.manual_seed(0)
        canvas = _make_canvas(capacity=20)
        canvas.initialize_random(seed=0)

        lr = torch.full((3, H, W), 0.5)
        # Identity spawn (matches what Gaussians already render to).
        spawn_match = GaussianBatch(
            xy=torch.rand(20, 2) * torch.tensor([W, H]),
            scale=torch.full((20, 2), 4.0),
            rot=torch.zeros(20),
            feat=torch.full((20, 3), 0.5),
        )
        max_prune = max(1, int(canvas.policy.max_prune_per_frame_frac
                               * canvas.capacity))
        for _ in range(5):
            canvas.update(motion=_zero_motion(), lr_frame=lr,
                          new_gaussians=spawn_match)
            assert canvas.capacity - max_prune <= canvas.n_alive <= canvas.capacity

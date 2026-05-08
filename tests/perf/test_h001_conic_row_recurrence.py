"""Validate H001 — conic row-recurrence math identity.

Hypothesis: for an EWA Gaussian with conic Λ = [[a,b],[b,d]] and quadratic
form q(x,y) = a·dx² + 2b·dx·dy + d·dy², the per-pixel weight w(x,y) = exp(-q/2)
along a fixed scanline can be marched with constant second difference.

    Δq_x  = q(x+1,y) - q(x,y) = a(2·dx + 1) + 2b·dy
    Δ²q_x = 2a   (constant)

    w_x = exp(-q_x/2)
    r_x = exp(-Δq_x/2)

    w_{x+1} = w_x · r_x
    r_{x+1} = r_x · exp(-a)

Cost: 2 exponentials per Gaussian-row, then pure FMAs across pixels.
vs naïve: 1 exponential per Gaussian-pixel pair.

This test validates the math identity is bit-exact within float rounding
BEFORE we port the optimization to the CUDA kernel. Math bugs found here
save days of CUDA debugging.

Reference: docs/research/hypotheses/H001-conic-row-recurrence.md
"""
from __future__ import annotations

import numpy as np
import pytest


def naive_q(a: float, b: float, d: float, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Direct quadratic form per pixel: q(x,y) = a·dx² + 2b·dx·dy + d·dy²."""
    return a * dx * dx + 2.0 * b * dx * dy + d * dy * dy


def naive_weights(
    a: float, b: float, d: float,
    cx: float, cy: float,
    pix_x: np.ndarray, pix_y: np.ndarray,
) -> np.ndarray:
    """Naive per-pixel exp(-q/2). Reference implementation."""
    dx = pix_x - cx
    dy = pix_y - cy
    q = naive_q(a, b, d, dx, dy)
    return np.exp(-0.5 * q)


def recurrence_weights_row(
    a: float, b: float, d: float,
    cx: float, cy: float,
    x_start: float, x_end: float, y_row: float,
) -> np.ndarray:
    """Row-recurrence weights for integer x ∈ [x_start, x_end] at fixed y_row.

    Compute one q₀, w₀ = exp(-q₀/2) at x_start, then march:
        Δq_x = a(2·dx + 1) + 2b·dy   evaluated at dx = dx_start
        r_x  = exp(-Δq_x/2)
        s    = exp(-a)               (constant)

        w_{x+1} = w_x · r_x
        r_{x+1} = r_x · s
    """
    n_x = int(x_end - x_start) + 1
    if n_x <= 0:
        return np.zeros(0, dtype=np.float64)

    dy = y_row - cy
    dx0 = x_start - cx

    q0 = a * dx0 * dx0 + 2.0 * b * dx0 * dy + d * dy * dy
    delta_q0 = a * (2.0 * dx0 + 1.0) + 2.0 * b * dy
    s = np.exp(-a)

    w = np.zeros(n_x, dtype=np.float64)
    w[0] = np.exp(-0.5 * q0)
    if n_x == 1:
        return w

    r = np.exp(-0.5 * delta_q0)
    for i in range(1, n_x):
        w[i] = w[i - 1] * r
        r = r * s

    return w


def recurrence_weights_tile(
    a: float, b: float, d: float,
    cx: float, cy: float,
    x0: int, y0: int, tile: int,
) -> np.ndarray:
    """Apply row recurrence per row across a (tile × tile) integer grid."""
    out = np.zeros((tile, tile), dtype=np.float64)
    for j in range(tile):
        out[j] = recurrence_weights_row(
            a, b, d, cx, cy,
            float(x0), float(x0 + tile - 1), float(y0 + j),
        )
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_recurrence_matches_naive_random_gaussians(seed: int) -> None:
    """For 100 random conic + center configurations, recurrence weights match
    naïve exp evaluation within fp64 rounding error."""
    rng = np.random.default_rng(seed)
    tile = 16
    x0, y0 = 0, 0
    pix_x_grid, pix_y_grid = np.meshgrid(
        np.arange(x0, x0 + tile, dtype=np.float64),
        np.arange(y0, y0 + tile, dtype=np.float64),
    )

    max_abs_err = 0.0
    for trial in range(100):
        # Sample a positive-definite conic.
        sx = rng.uniform(0.5, 4.0)
        sy = rng.uniform(0.5, 4.0)
        theta = rng.uniform(0.0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        a = c * c / (sx * sx) + s * s / (sy * sy)
        d = s * s / (sx * sx) + c * c / (sy * sy)
        b = c * s * (1.0 / (sx * sx) - 1.0 / (sy * sy))
        # Center inside the tile, sub-pixel
        cx = rng.uniform(0.0, tile)
        cy = rng.uniform(0.0, tile)

        ref = naive_weights(a, b, d, cx, cy, pix_x_grid, pix_y_grid)
        rec = recurrence_weights_tile(a, b, d, cx, cy, x0, y0, tile)

        err = float(np.max(np.abs(ref - rec)))
        max_abs_err = max(max_abs_err, err)

    # Identity should be exact within fp64 rounding accumulation over 16-pixel
    # scanlines. Empirically the worst case is dominated by the s = exp(-a)
    # constant-multiply growth, which stays well below 1e-12 at a < 4.
    assert max_abs_err < 1e-10, f"H001 row-recurrence drift exceeds 1e-10: {max_abs_err:.3e}"


def test_recurrence_constant_second_difference() -> None:
    """Δ²q_x = 2a should hold exactly for any conic + center."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(0.05, 4.0)
        b = rng.uniform(-1.0, 1.0)
        d = rng.uniform(0.05, 4.0)
        cx, cy = rng.uniform(0.0, 16.0, size=2)
        y_row = rng.uniform(0.0, 16.0)

        # Compute Δq_x at three consecutive integer x and check Δ²q_x = 2a.
        dy = y_row - cy
        dq = lambda dx: a * (2.0 * dx + 1.0) + 2.0 * b * dy
        for x in [-3, 0, 5, 12]:
            dx = float(x) - cx
            d2q = dq(dx + 1.0) - dq(dx)
            assert abs(d2q - 2.0 * a) < 1e-12, (
                f"Δ²q_x mismatch at x={x}: got {d2q}, expected {2.0 * a}"
            )


def test_recurrence_preserves_isotropic_symmetry() -> None:
    """For an isotropic Gaussian (a=d, b=0) centered in tile, recurrence
    output should be symmetric across the column axis."""
    a = 0.5
    b = 0.0
    d = 0.5
    tile = 16
    cx = (tile - 1) / 2.0
    cy = cx
    rec = recurrence_weights_tile(a, b, d, cx, cy, 0, 0, tile)
    # Row j should mirror around column cx
    for j in range(tile):
        row = rec[j]
        assert np.allclose(row, row[::-1], atol=1e-12), (
            f"row {j} not symmetric for isotropic Gaussian"
        )


def test_recurrence_no_negative_drift() -> None:
    """Weights must remain non-negative after recurrence; multiplicative
    march cannot produce negatives unless an underflow→denormal→0 occurs."""
    rng = np.random.default_rng(1)
    tile = 32  # Wider tile to stress drift
    for _ in range(20):
        a = rng.uniform(0.05, 0.5)  # Smaller a → less aggressive decay
        b = rng.uniform(-0.2, 0.2)
        d = rng.uniform(0.05, 0.5)
        cx, cy = rng.uniform(0.0, float(tile)), rng.uniform(0.0, float(tile))
        rec = recurrence_weights_tile(a, b, d, cx, cy, 0, 0, tile)
        assert (rec >= 0).all(), "recurrence produced negative weights"


# ---------------------------------------------------------------------------
# CUDA-readiness summary (for human readers)
# ---------------------------------------------------------------------------
#
# Math identity confirmed by these tests at fp64. Next step for porting to
# CUDA (rasterizer_fwd.cu inner loop):
#
#   1. At each tile-row entry, compute q0 (using existing dx/dy precompute),
#      w0 = __expf(-0.5f * q0), r0 = __expf(-0.5f * delta_q0), s = __expf(-a).
#   2. March across the row in the WMMA weight setup (lines 203-227) using
#      multiplies instead of recomputing q + expf per pixel.
#   3. Bit-equivalence at fp32 within tol ~ 1e-5; expected speedup 2-4×
#      on the rasterizer arithmetic-bound paths.
#
# Risks for CUDA port:
#   - Cumulative float drift over scanlines >32 px (we tile-bound naturally,
#     so this should not occur).
#   - Register pressure increase from holding (w, r, s) across the loop.
#     Compare nvcc register spill output before/after.
#   - Dispatch overhead if we conditionally select between recurrence and
#     LUT kernels per Gaussian. Profile.
#
# DO NOT modify rasterizer_fwd.cu without:
#   1. This test passing
#   2. A CUDA test matching the recurrence kernel against the existing one
#      within fp32 tol on real Gaussian batches
#   3. A microbench showing >=1.5× speedup on the conic-eval-bound path

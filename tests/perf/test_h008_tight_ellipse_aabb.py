"""Validate H008 — tight ellipse AABB for rotated anisotropic Gaussians.

For a Gaussian with conic (a, b, d) where a*d - b² > 0:

    r_x = sqrt(tau * d / (a*d - b*b))
    r_y = sqrt(tau * a / (a*d - b*b))

at the τ-Mahalanobis isocontour q = τ. For τ = 9, this is the 3σ ellipse.

Test: verify the formula gives correct per-axis extent, including the
hardest case (rotated anisotropic where conservative `3*max(sx,sy)` greatly
over-estimates) and the easy case (axis-aligned where it matches `3*sx, 3*sy`).
"""
from __future__ import annotations

import numpy as np
import pytest


def conic_from_scale_rot(sx: float, sy: float, theta: float) -> tuple[float, float, float]:
    c, s = np.cos(theta), np.sin(theta)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    a = c * c * inv_sx2 + s * s * inv_sy2
    d = s * s * inv_sx2 + c * c * inv_sy2
    b = c * s * (inv_sx2 - inv_sy2)
    return a, b, d


def tight_aabb(a: float, b: float, d: float, tau: float = 9.0) -> tuple[float, float]:
    det = a * d - b * b
    assert det > 0, f"singular conic: a={a}, b={b}, d={d}"
    r_x = np.sqrt(tau * d / det)
    r_y = np.sqrt(tau * a / det)
    return r_x, r_y


def conservative_aabb(sx: float, sy: float) -> float:
    return 3.0 * max(sx, sy)


def q_from_dx_dy(a: float, b: float, d: float, dx: float, dy: float) -> float:
    return a * dx * dx + 2.0 * b * dx * dy + d * dy * dy


def test_axis_aligned_matches_3sigma() -> None:
    """For axis-aligned Gaussian (theta=0), tight should equal 3*sx, 3*sy."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        sx = rng.uniform(0.5, 4.0)
        sy = rng.uniform(0.5, 4.0)
        a, b, d = conic_from_scale_rot(sx, sy, 0.0)
        r_x, r_y = tight_aabb(a, b, d)
        assert abs(r_x - 3.0 * sx) < 1e-6, f"axis-aligned x mismatch: {r_x} vs {3.0 * sx}"
        assert abs(r_y - 3.0 * sy) < 1e-6, f"axis-aligned y mismatch: {r_y} vs {3.0 * sy}"


def test_isotropic_matches_3sigma_after_rotation() -> None:
    """Isotropic Gaussian (sx=sy) should have rotation-invariant AABB."""
    s = 1.5
    a0, b0, d0 = conic_from_scale_rot(s, s, 0.0)
    r0 = tight_aabb(a0, b0, d0)
    for theta in np.linspace(0, np.pi, 7):
        a, b, d = conic_from_scale_rot(s, s, theta)
        r = tight_aabb(a, b, d)
        assert abs(r[0] - r0[0]) < 1e-6
        assert abs(r[1] - r0[1]) < 1e-6


def test_rotated_anisotropic_tight_beats_conservative() -> None:
    """Key claim: for rotated anisotropic Gaussians, tight AABB is meaningfully
    smaller than conservative `3*max(sx, sy)` along the short axis."""
    sx, sy = 1.0, 4.0  # 4:1 anisotropic
    # Rotate by pi/2 → the long axis should now be along x, NOT y.
    a, b, d = conic_from_scale_rot(sx, sy, np.pi / 2.0)
    r_x, r_y = tight_aabb(a, b, d)
    cons = conservative_aabb(sx, sy)  # = 3 * 4 = 12

    # After 90° rotation: a Gaussian that was 1 wide and 4 tall becomes 4 wide, 1 tall.
    # So r_x ≈ 12 (3*4), r_y ≈ 3 (3*1).
    assert abs(r_x - 12.0) < 1e-5, f"r_x after 90deg rotation expected ~12, got {r_x}"
    assert abs(r_y - 3.0) < 1e-5, f"r_y after 90deg rotation expected ~3, got {r_y}"

    # Conservative would use 12 for BOTH axes. Tight gives 12, 3.
    # Tight saves area = (12*12 - 12*3) / (12*12) = 75%.
    cons_area = (2 * cons) ** 2
    tight_area = (2 * r_x) * (2 * r_y)
    savings = 1.0 - tight_area / cons_area
    assert savings > 0.7, f"tight should save ~75% area for 4:1 90deg, got {savings:.2%}"


def test_aabb_contains_3sigma_isocontour() -> None:
    """The tight AABB at τ=9 must contain every (dx, dy) where q ≤ 9."""
    rng = np.random.default_rng(1)
    for trial in range(50):
        sx = rng.uniform(0.5, 3.0)
        sy = rng.uniform(0.5, 3.0)
        theta = rng.uniform(0, np.pi)
        a, b, d = conic_from_scale_rot(sx, sy, theta)
        r_x, r_y = tight_aabb(a, b, d, tau=9.0)

        # Sample the τ=9 ellipse boundary parametrically; every point should
        # fall inside the AABB [-r_x, r_x] × [-r_y, r_y].
        for t in np.linspace(0, 2 * np.pi, 64):
            # Unit circle in conic space → ellipse in (dx, dy) via inverse
            # eigendecomp; easier to just verify that any point on q=9 is
            # bounded by the AABB.
            # Use polar sweep over dx-dy at fixed q=9: parametrize by direction θ:
            # dx = ρ cosθ, dy = ρ sinθ where ρ² (a cos²θ + 2b sinθcosθ + d sin²θ) = 9
            ct, st = np.cos(t), np.sin(t)
            denom = a * ct * ct + 2 * b * ct * st + d * st * st
            if denom <= 0:
                continue
            rho = np.sqrt(9.0 / denom)
            dx = rho * ct
            dy = rho * st
            # Verify q = 9 (within rounding)
            q = q_from_dx_dy(a, b, d, dx, dy)
            assert abs(q - 9.0) < 1e-4
            # Verify (dx, dy) inside AABB (with tiny epsilon for fp rounding)
            eps = 1e-4
            assert abs(dx) <= r_x + eps, (
                f"trial {trial}: ellipse boundary point dx={dx} exceeds r_x={r_x}"
            )
            assert abs(dy) <= r_y + eps, (
                f"trial {trial}: ellipse boundary point dy={dy} exceeds r_y={r_y}"
            )


def test_aabb_corner_is_inside_or_on_3sigma() -> None:
    """The tight AABB corner (r_x, r_y) must lie OUTSIDE the 3σ ellipse —
    i.e., q(r_x, r_y) >= τ. This validates the AABB doesn't crop the
    ellipse, only bounds it."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        sx = rng.uniform(0.5, 3.0)
        sy = rng.uniform(0.5, 3.0)
        theta = rng.uniform(0, np.pi)
        a, b, d = conic_from_scale_rot(sx, sy, theta)
        r_x, r_y = tight_aabb(a, b, d, tau=9.0)

        # Corner (r_x, r_y): only on the ellipse if both axes are extremal,
        # which is only true for axis-aligned Gaussians. For rotated, the
        # corner is OUTSIDE the ellipse — i.e., q(r_x, r_y) > 9. Either way:
        # the corner must NOT be strictly inside (q < 9).
        q_corner = q_from_dx_dy(a, b, d, r_x, r_y)
        assert q_corner >= 9.0 - 1e-3, (
            f"AABB corner inside 3-sigma ellipse: q={q_corner} for sx={sx}, sy={sy}, theta={theta}"
        )


def test_savings_summary() -> None:
    """Quantify expected area savings for typical workloads; informs the
    'reduces tile-list length' performance claim."""
    # Random anisotropic Gaussians at random rotations, typical scales.
    rng = np.random.default_rng(42)
    n = 1000
    cons_areas = np.zeros(n)
    tight_areas = np.zeros(n)
    for i in range(n):
        sx = rng.uniform(0.5, 3.0)
        sy = rng.uniform(0.5, 3.0)
        theta = rng.uniform(0, np.pi)
        a, b, d = conic_from_scale_rot(sx, sy, theta)
        r_x, r_y = tight_aabb(a, b, d)
        cons = conservative_aabb(sx, sy)
        cons_areas[i] = (2 * cons) ** 2
        tight_areas[i] = (2 * r_x) * (2 * r_y)
    avg_savings = 1.0 - tight_areas.mean() / cons_areas.mean()
    median_ratio = float(np.median(tight_areas / cons_areas))
    # We expect ≥30% area savings on average for diverse anisotropy/rotation
    assert avg_savings > 0.3, f"insufficient avg savings: {avg_savings:.2%}"
    print(f"H008 area savings: avg={avg_savings:.1%}, median ratio={median_ratio:.3f}")

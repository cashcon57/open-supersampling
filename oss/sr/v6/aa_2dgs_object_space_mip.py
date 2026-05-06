"""AA-2DGS object-space Mip filter (per-splat density modulation).

Source: Younes & Boukhayma, "Anti-Aliased 2D Gaussian Splatting,"
NeurIPS 2025. arXiv:2506.11252.

The published formulation derives an object-space Mip filter via an
affine approximation of the ray-splat intersection mapping: the screen-
space pixel footprint is expressed in the splat's local 2D frame, and
the integral of the splat's density over that footprint produces a
per-splat scale factor that modulates opacity for the current view.

The closed form for "1D Gaussian convolved with a unit box" is

    integral over [-w/2, w/2] of N(0, sigma) dx = erf(w / (2*sqrt(2)*sigma))

For a per-splat Mip factor we want the total mass under a box of width
``w = target_pixel_size / base_pixel_size`` (the ratio of the current
target footprint to the splat's training-time reference footprint),
applied separably on each principal axis. When ``w`` is large
(zoom-out / minification) the Gaussian is poorly resolved by the
sampling grid and the convolved mass is *less* than 1, so opacity is
reduced -- matching a downsampled view. When ``w`` is small (zoom-in)
the result trends to 1 and opacity is preserved or gently boosted.

For our reference we collapse the per-axis ratio to a single scalar
because the V6 caller passes a single ``target_pixel_size`` per frame.
The CUDA kernel is expected to extend this to a directional ratio
(footprint Jacobian's two singular values).

NOTE: PyTorch pure-functional reference implementation. Slow but
correct. Production CUDA kernels follow as a separate sprint.
"""

from __future__ import annotations

import math

import torch

__all__ = ["object_space_mip_factor"]

# 1 / sqrt(2) for the erf argument scaling.
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def object_space_mip_factor(
    gaussian_scales: torch.Tensor,
    target_pixel_size: float,
    base_pixel_size: float = 1.0,
) -> torch.Tensor:
    """Per-Gaussian Mip-filter density modulation factor.

    Reference: AA-2DGS object-space filter, abstracted to the closed-form
    integral of a 2D Gaussian convolved with a unit pixel box. For each
    Gaussian with principal-axis std-devs ``(s1, s2)`` and a target
    footprint of width ``w = target_pixel_size / base_pixel_size``, the
    per-axis convolved mass is

        m_i = erf( w / (2 * sqrt(2) * s_i) )

    and the joint factor is ``m_1 * m_2``. When ``w == 0`` the factor
    is 0 (degenerate) -- caller is expected to pass strictly positive
    pixel sizes. When ``s_i`` is very small relative to ``w`` the factor
    saturates at 1 (Gaussian fully contained inside the pixel box).
    When ``s_i`` is very large relative to ``w`` the factor goes to
    ``w / (sqrt(2*pi) * s_i)`` -- the area of the box at peak density --
    which correctly attenuates the splat.

    The factor returned here is the multiplier that the caller multiplies
    into per-Gaussian opacity for the current view. ``base_pixel_size``
    is the splat's training-time reference pixel size (typically 1.0,
    in pixel units of the training resolution); ``target_pixel_size``
    is the current view's pixel footprint at the splat's depth, in the
    same units.

    Args:
        gaussian_scales:    (N, 2) per-Gaussian 2D principal-axis std-devs
            (i.e. sqrt of eigenvalues of Sigma in object space).
        target_pixel_size:  current target pixel footprint at this splat's
            depth, in object-space units (matching gaussian_scales).
        base_pixel_size:    splat's training-time reference pixel size
            in the same units. Default 1.0.

    Returns:
        (N,) per-Gaussian opacity multiplier in (0, 1].

    Notes:
        * Pure-functional, broadcasts over (N,).
        * bf16-safe (uses torch.erf, which is autocast-stable).
        * Degenerate ``s_i = 0`` (line-Gaussian) is clamped to a small
          floor so erf is well-defined; the floor maps to factor ~1
          (the splat is so thin the pixel box always contains it).
    """
    if gaussian_scales.ndim < 1 or gaussian_scales.shape[-1] != 2:
        raise ValueError(
            f"gaussian_scales must be (..., 2), got {tuple(gaussian_scales.shape)}"
        )
    if target_pixel_size <= 0.0:
        raise ValueError(
            f"target_pixel_size must be > 0, got {target_pixel_size}"
        )
    if base_pixel_size <= 0.0:
        raise ValueError(
            f"base_pixel_size must be > 0, got {base_pixel_size}"
        )

    # Pixel-box width in object-space units.
    w = float(target_pixel_size) / float(base_pixel_size)

    # erf arg per axis: w / (2 * sqrt(2) * s_i)
    # Floor s_i to avoid 0-division on degenerate splats. The floor is
    # well below any meaningful std-dev; at s_i << w the erf argument
    # blows up and erf saturates at 1, which is the right limit.
    s = torch.clamp(gaussian_scales, min=1.0e-12)
    arg = (w * 0.5 * _INV_SQRT_2) / s  # (..., 2)
    m = torch.erf(arg)  # (..., 2)

    # Joint per-Gaussian factor: product of the two per-axis masses.
    return m[..., 0] * m[..., 1]

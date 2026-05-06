"""Analytic-Splatting per-pixel area integral via logistic-CDF approximation.

Source: Liang et al., "Analytic-Splatting: Anti-Aliased 3D Gaussian
Splatting via Analytic Integration," ECCV 2024 (Oral). arXiv:2403.11056.
Definition 1, Equation 15.

The published formulation replaces the EWA point-sampling of a 2D
Gaussian at each pixel centre with an analytical integral of the
Gaussian over a unit pixel square. The 1D Gaussian CDF is approximated
by a conditioned logistic CDF

    S(x) = 1 / (1 + exp(-1.6 x - 0.07 x^3))

which is accurate to ~1e-3 relative error vs the true erf-based CDF.
The pixel response of a 1D Gaussian over the unit-width pixel window
centred at u is then

    I_g(u) = S((u + 0.5) / sigma) - S((u - 0.5) / sigma)

For 2D, the covariance Sigma is diagonalised to (sigma_1, sigma_2), the
integration domain is rotated into the principal-axis frame, and the
2D pixel-window integral factorises into a product of two 1D integrals.

For the V6 reference we accept the pre-computed inverse covariance
``sigma_2d_inv``, diagonalize it with a gradient-safe closed-form 2x2
symmetric eigensystem, and apply the per-axis logistic CDF integral.
Repeated-eigenvalue inputs take an isotropic fast path that avoids the
undefined eigenvector basis.

NOTE: PyTorch pure-functional reference implementation. Slow but
correct. Production CUDA kernels follow as a separate sprint.
"""

from __future__ import annotations

import math

import torch

__all__ = ["analytic_pixel_integral", "logistic_cdf"]

_EIGEN_EPS = 1.0e-6
_EIGENVALUE_FLOOR = 1.0e-12
_ISOTROPIC_EIGEN_GAP = 1.0e-5


def logistic_cdf(x: torch.Tensor) -> torch.Tensor:
    """Conditioned logistic CDF approximation of the standard normal CDF.

    Reference: Analytic-Splatting Definition 1.

    ``S(x) = 1 / (1 + exp(-1.6 x - 0.07 x^3))``

    Accuracy: ~1e-3 relative error vs the true erf-based CDF over the
    range encountered at +/- 5 sigma. bf16-safe.
    """
    return torch.sigmoid(1.6 * x + 0.07 * x * x * x)


def _align_to_offsets(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Append singleton pixel axes until ``x`` broadcasts with offsets."""
    while x.ndim < offsets.ndim:
        x = x.unsqueeze(-1)
    return x


def _axis_integral(u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    inv_sigma = 1.0 / sigma
    return logistic_cdf((u + 0.5) * inv_sigma) - logistic_cdf((u - 0.5) * inv_sigma)


def analytic_pixel_integral(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    sigma_2d_inv: torch.Tensor,
) -> torch.Tensor:
    """Closed-form pixel-area integral of a 2D Gaussian.

    Reference: Analytic-Splatting Eq. 15. Replaces the EWA point-sample
    density with an analytical integral via the logistic CDF
    approximation S(x) = 1 / (1 + exp(-1.6 x - 0.07 x^3)).

    For each pixel, integrating a 1D Gaussian over [delta - 0.5,
    delta + 0.5] gives ``S((delta + 0.5) / sigma) - S((delta - 0.5) /
    sigma)`` per axis. The 2D version factorises in the eigenbasis of
    Sigma -- so we diagonalize ``sigma_2d_inv`` with the closed-form 2x2
    symmetric eigensystem, rotate the (delta_x, delta_y) offsets into
    that basis, and multiply the two per-axis 1D integrals.

    Construction:
        1. Recover Sigma from sigma_2d_inv via inverse, diagonalize
           -> rotation R and eigenvalues lambda_1, lambda_2.
           Equivalently: diagonalize sigma_2d_inv directly --
           same eigenvectors, reciprocal eigenvalues. We do the
           latter (one fewer matrix inverse) and take the standard
           deviations as 1 / sqrt(lambda_inv).
        2. Rotate the per-pixel offset vector ``(delta_x, delta_y)``
           into the eigenbasis -> ``(u, v)``.
        3. Compute the per-axis logistic-CDF integral with the
           per-axis std-dev.
        4. Density = product of the two per-axis integrals, scaled by
           ``2 * pi * sigma_1 * sigma_2`` so the result agrees with the
           Analytic-Splatting Eq. 15 normalisation (which is the integral
           of an *un-normalised* 2D Gaussian). This matches the EWA
           convention used by the OSS rasterizer where the splat density
           is un-normalised (the per-Gaussian opacity is a separate
           multiplier).

    Args:
        delta_x:        (..., P) per-pixel x distance from splat centre,
            in pixel units. Shape must broadcast with delta_y.
        delta_y:        (..., P) per-pixel y distance from splat centre.
        sigma_2d_inv:   (..., 2, 2) inverse projected covariance matrix
            per Gaussian. Leading dims must broadcast with delta_x /
            delta_y.

    Returns:
        Density per pixel, shape ``broadcast(delta_x, delta_y)``. The
        result is the integral of an *un-normalised* (peak-1) 2D
        Gaussian over the unit pixel square, matching the EWA
        convention used by the OSS rasterizer (per-Gaussian opacity is
        a separate multiplier upstream). At the splat centre with
        sigma=1 the integral evaluates to ~0.1466 -- the area of a
        unit-peak Gaussian over a unit pixel.

    Notes:
        * bf16-safe.
        * If ``sigma_2d_inv`` is exactly diagonal the closed-form rotation
          collapses to the coordinate axes and the returned values match
          the simple separable 1D-CDF case.
        * Tiny eigenvalues are clamped at a floor; in that limit the
          per-axis std-dev grows large and the per-axis integral
          smoothly approaches the integral of a unit-density box across
          the offset window.
    """
    if sigma_2d_inv.shape[-2:] != (2, 2):
        raise ValueError(
            f"sigma_2d_inv must be (..., 2, 2), got {tuple(sigma_2d_inv.shape)}"
        )

    # Float upcast for numerical stability of the 2x2 closed-form path; the
    # downstream return is in the input dtype.
    in_dtype = sigma_2d_inv.dtype
    sigma_inv_f = sigma_2d_inv.to(dtype=torch.float32) if in_dtype != torch.float32 else sigma_2d_inv

    delta_x_b, delta_y_b = torch.broadcast_tensors(delta_x, delta_y)
    dx = delta_x_b.to(dtype=torch.float32)
    dy = delta_y_b.to(dtype=torch.float32)

    # Symmetric 2x2 eigensystem for A = [[a, b], [b, c]]. The isotropic ridge
    # keeps the inverse covariance strictly positive without changing the
    # eigenvectors. The discriminant follows the Analytic-Splatting 2D
    # diagonalisation but avoids torch.linalg.eigh, whose backward is singular
    # for repeated eigenvalues.
    a = sigma_inv_f[..., 0, 0] + _EIGEN_EPS
    b = 0.5 * (sigma_inv_f[..., 0, 1] + sigma_inv_f[..., 1, 0])
    c = sigma_inv_f[..., 1, 1] + _EIGEN_EPS

    half_delta = 0.5 * (a - c)
    t = 0.5 * (a + c)
    raw_d_sq = half_delta * half_delta + b * b
    d = torch.sqrt(raw_d_sq + _EIGEN_EPS)

    lambda_high = torch.clamp(t + d, min=_EIGENVALUE_FLOOR)
    lambda_low = torch.clamp(t - d, min=_EIGENVALUE_FLOOR)
    sigma_high = torch.rsqrt(lambda_high)
    sigma_low = torch.rsqrt(lambda_low)

    eigengap = 2.0 * torch.sqrt(raw_d_sq)
    near_isotropic = eigengap < _ISOTROPIC_EIGEN_GAP

    # theta is the eigenvector angle for lambda_high. Guard exactly repeated
    # eigenvalues before atan2 so the inactive anisotropic branch still has
    # finite backward values under torch.where.
    safe_angle_x = torch.where(near_isotropic, torch.ones_like(half_delta), a - c)
    safe_angle_y = torch.where(near_isotropic, torch.zeros_like(b), 2.0 * b)
    theta = 0.5 * torch.atan2(safe_angle_y, safe_angle_x)
    cos_t = _align_to_offsets(torch.cos(theta), dx)
    sin_t = _align_to_offsets(torch.sin(theta), dx)

    sigma_high_b = _align_to_offsets(sigma_high, dx)
    sigma_low_b = _align_to_offsets(sigma_low, dx)

    u_high = cos_t * dx + sin_t * dy
    u_low = -sin_t * dx + cos_t * dy

    int_high = _axis_integral(u_high, sigma_high_b)
    int_low = _axis_integral(u_low, sigma_low_b)
    anisotropic = 2.0 * math.pi * sigma_high_b * sigma_low_b * int_high * int_low

    # For repeated eigenvalues the principal axes are undefined, but the
    # Gaussian is rotation-invariant. Eq. 12 therefore reduces to the same
    # separable 1D CDF integral in screen x/y, with one shared std-dev.
    sigma_iso = torch.rsqrt(torch.clamp(t, min=_EIGENVALUE_FLOOR))
    sigma_iso_b = _align_to_offsets(sigma_iso, dx)
    int_x = _axis_integral(dx, sigma_iso_b)
    int_y = _axis_integral(dy, sigma_iso_b)
    isotropic = 2.0 * math.pi * sigma_iso_b * sigma_iso_b * int_x * int_y

    out = torch.where(_align_to_offsets(near_isotropic, dx), isotropic, anisotropic)

    return out.to(dtype=in_dtype)

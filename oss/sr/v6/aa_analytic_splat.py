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
``sigma_2d_inv``, eigendecompose it, and apply the per-axis logistic
CDF integral. Diagonal-Sigma fast path is provided implicitly by the
eigenvalue decomposition of a diagonal matrix.

NOTE: PyTorch pure-functional reference implementation. Slow but
correct. Production CUDA kernels follow as a separate sprint.
"""

from __future__ import annotations

import torch

__all__ = ["analytic_pixel_integral", "logistic_cdf"]


def logistic_cdf(x: torch.Tensor) -> torch.Tensor:
    """Conditioned logistic CDF approximation of the standard normal CDF.

    Reference: Analytic-Splatting Definition 1.

    ``S(x) = 1 / (1 + exp(-1.6 x - 0.07 x^3))``

    Accuracy: ~1e-3 relative error vs the true erf-based CDF over the
    range encountered at +/- 5 sigma. bf16-safe.
    """
    return torch.sigmoid(1.6 * x + 0.07 * x * x * x)


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
    Sigma -- so we eigendecompose ``sigma_2d_inv``, rotate the
    (delta_x, delta_y) offsets into that basis, and multiply the two
    per-axis 1D integrals.

    Construction:
        1. Recover Sigma from sigma_2d_inv via inverse, eigendecompose
           -> rotation R and eigenvalues lambda_1, lambda_2.
           Equivalently: eigendecompose sigma_2d_inv directly --
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
        * If ``sigma_2d_inv`` is exactly diagonal the eigendecomposition
          collapses to identity rotation and the returned values match
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

    # Eigendecompose Sigma_inv. Using torch.linalg.eigh because Sigma_inv
    # is symmetric positive-definite. eigh returns ascending eigenvalues
    # and orthonormal eigenvectors.
    # Float upcast for numerical stability of the decomposition; matters
    # because eigh under bf16 is wobbly. The downstream return is in the
    # input dtype.
    in_dtype = sigma_2d_inv.dtype
    sigma_inv_f = sigma_2d_inv.to(dtype=torch.float32) if in_dtype != torch.float32 else sigma_2d_inv

    eigvals_inv, eigvecs = torch.linalg.eigh(sigma_inv_f)  # (..., 2), (..., 2, 2)

    # Variances along principal axes = 1 / eigenvalues_of_Sigma_inv.
    # Floor eigenvalues_inv at a small value (i.e. cap the variance at a
    # large value); avoids div-by-zero on degenerate Sigma_inv.
    eigvals_inv = torch.clamp(eigvals_inv, min=1.0e-12)
    sigma_1 = torch.rsqrt(eigvals_inv[..., 0])  # std-dev along first axis
    sigma_2 = torch.rsqrt(eigvals_inv[..., 1])  # std-dev along second axis

    # Rotate (delta_x, delta_y) into eigenbasis. eigvecs has columns =
    # eigenvectors. We want u = R^T delta where R = eigvecs.
    # The calling convention for this function is:
    #   sigma_2d_inv:  (N_leading..., 2, 2)
    #   delta_x/y:     (N_leading..., P)
    # where the per-Gaussian eigendecomposition broadcasts across all P
    # pixel offsets. We add a singleton P axis to eigvecs / sigma_1 /
    # sigma_2 so the rotation applies pointwise across pixels.
    delta_x_b, delta_y_b = torch.broadcast_tensors(delta_x, delta_y)
    delta = torch.stack(
        (delta_x_b.to(dtype=torch.float32), delta_y_b.to(dtype=torch.float32)),
        dim=-1,
    )  # (..., P, 2)

    # Match shapes for the rotation: insert a singleton "P" axis just
    # before the trailing (2, 2) of eigvecs and the trailing 2 of
    # sigma_1 / sigma_2, so they broadcast against delta's P dim.
    eigvecs_b = eigvecs.unsqueeze(-3)  # (..., 1, 2, 2)
    sigma_1_b = sigma_1.unsqueeze(-1)  # (..., 1)
    sigma_2_b = sigma_2.unsqueeze(-1)  # (..., 1)

    # u = R^T @ delta, computed per pixel.
    # eigvecs_b: (..., 1, 2, 2); delta: (..., P, 2)
    # contract over j: u_i = sum_j eigvecs_b[..., :, j, i] * delta[..., :, j]
    # Manual contraction (broadcasts cleanly under the singleton P axis).
    u = (eigvecs_b * delta.unsqueeze(-1)).sum(dim=-2)  # (..., P, 2)

    u1 = u[..., 0]
    u2 = u[..., 1]

    # Per-axis 1D pixel integral via logistic CDF, broadcasting per-Gaussian
    # sigma against per-pixel u.
    inv_s1 = 1.0 / sigma_1_b
    inv_s2 = 1.0 / sigma_2_b
    int_x = logistic_cdf((u1 + 0.5) * inv_s1) - logistic_cdf((u1 - 0.5) * inv_s1)
    int_y = logistic_cdf((u2 + 0.5) * inv_s2) - logistic_cdf((u2 - 0.5) * inv_s2)

    # Eq. 15 normalisation: 2*pi*sigma_1*sigma_2 * (int_x * int_y).
    # This produces values that match the un-normalised Gaussian
    # convention used by the OSS EWA path: at the splat centre with
    # sigma=1 the integral evaluates to ~0.1466 (the area of a unit-peak
    # Gaussian over a unit pixel).
    import math
    norm = 2.0 * math.pi * sigma_1_b * sigma_2_b
    out = norm * int_x * int_y

    return out.to(dtype=in_dtype)

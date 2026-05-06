"""GS-STVSR resampled output covariance.

Reference: Zhou et al., GS-STVSR (arXiv:2604.18047). The resampled
covariance for a 2D Gaussian transformed from a source resolution into
a target resolution is

    Sigma'_output = J_t Sigma_t J_t^T + Sigma_recon

with J_t the 2x2 Jacobian of the spatial transformation (uniform 2x2
upscale composed with the per-pixel motion-warp Jacobian for v6) and
Sigma_recon a fixed reconstruction-filter covariance pinned to the
target resolution. Adding Sigma_recon is what makes the EWA filter
anti-aliasing happen by construction.

Pure functions, no module state. Broadcasts over leading dims.
"""
from __future__ import annotations

import torch

# Isotropic ridge added to (J Sigma J^T + Sigma_recon) to keep the result
# strictly positive-definite when callers pass degenerate Jacobians at
# warp / projection boundaries. Chosen well below typical Sigma_recon scales
# (~quarter-pixel^2 ~ 0.0625 in our canonical 2x case) so it never
# perceptibly shifts the well-conditioned regime, and well above bf16
# round-off (~6e-3 relative) to stay meaningful in low-precision paths.
# Gradient-safe (constant ridge), unlike an eigh-based clamp which produces
# NaN gradients on inputs with repeated eigenvalues.
_PD_RIDGE = 1.0e-7


def isotropic_sigma_recon(
    pixel_size: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Isotropic EWA reconstruction-filter covariance for a target pixel.

    Canonical choice: ``(pixel_size / 2)^2 * I``. The half-pixel std-dev
    matches a Gaussian whose -1 sigma extent reaches the pixel-center
    Nyquist limit at the target resolution.

    Args:
        pixel_size: target-resolution pixel edge length in source-resolution
            units (e.g. 0.5 when upsampling 2x — one HR pixel covers half an
            LR pixel).
        device: torch device.
        dtype: tensor dtype.

    Returns:
        ``(2, 2)`` covariance.
    """
    if pixel_size <= 0.0:
        raise ValueError(f"pixel_size must be positive; got {pixel_size}")
    half = 0.5 * float(pixel_size)
    var = half * half
    out = torch.zeros((2, 2), device=device, dtype=dtype)
    out[0, 0] = var
    out[1, 1] = var
    return out


def _ensure_positive_definite(sigma: torch.Tensor) -> torch.Tensor:
    """Symmetrize and add an isotropic ridge for PD safety.

    Operates on the symmetric part so callers don't need to symmetrize
    upstream. The ridge is gradient-safe; a true eigenvalue clamp via
    ``eigh`` would NaN-out the backward pass on inputs with repeated
    eigenvalues (e.g. isotropic Gaussians, which dominate at canvas init).
    """
    sym = 0.5 * (sigma + sigma.transpose(-1, -2))
    eye = torch.eye(2, device=sym.device, dtype=sym.dtype)
    return sym + _PD_RIDGE * eye


def resample_covariance(
    sigma_t: torch.Tensor,
    jacobian_t: torch.Tensor,
    sigma_recon: torch.Tensor | float,
) -> torch.Tensor:
    """GS-STVSR covariance resampling.

    Computes ``J Sigma J^T + Sigma_recon`` with eigenvalue clamping to keep
    the result positive-definite at warp boundaries (where ``J`` can collapse
    to near-singular).

    Args:
        sigma_t: ``(..., 2, 2)`` source-resolution spatial covariance.
        jacobian_t: ``(..., 2, 2)`` Jacobian of the spatial transform.
        sigma_recon: ``(..., 2, 2)`` reconstruction-filter covariance, or a
            scalar (treated as isotropic ``s * I``).

    Returns:
        ``(..., 2, 2)`` resampled output covariance.
    """
    if sigma_t.shape[-2:] != (2, 2):
        raise ValueError(f"sigma_t must end in (2, 2); got {tuple(sigma_t.shape)}")
    if jacobian_t.shape[-2:] != (2, 2):
        raise ValueError(f"jacobian_t must end in (2, 2); got {tuple(jacobian_t.shape)}")

    j = jacobian_t
    jt = j.transpose(-1, -2)
    transformed = j @ sigma_t @ jt

    if isinstance(sigma_recon, torch.Tensor):
        if sigma_recon.shape[-2:] != (2, 2):
            raise ValueError(
                f"sigma_recon tensor must end in (2, 2); got {tuple(sigma_recon.shape)}"
            )
        recon = sigma_recon.to(device=transformed.device, dtype=transformed.dtype)
    else:
        scalar = float(sigma_recon)
        recon = torch.eye(2, device=transformed.device, dtype=transformed.dtype) * scalar

    out = transformed + recon
    return _ensure_positive_definite(out)


__all__ = ["isotropic_sigma_recon", "resample_covariance"]

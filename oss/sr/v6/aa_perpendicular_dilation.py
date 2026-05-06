"""AAA-Gaussians perpendicular-ray covariance dilation (2D adaptation).

Source: Steiner et al., "AAA-Gaussians: Anti-Aliased and Artifact-Free 3D
Gaussian Rendering," ICCV 2025 (Highlight). arXiv:2504.12811. Equation 10.

The published formulation derives, for a rank-3 covariance Sigma in world
space, a per-frame dilation that adds a fixed worst-case sampling-rate
variance only along the components of Sigma perpendicular to the viewing
ray d. The along-d component is left untouched, which avoids the
over-transparency that an isotropic Mip-Splatting dilation produces under
wide-FOV / near-camera viewing.

OSS uses 2D Gaussians (rank-2 disks already projected to the image
plane). For the 2D-disk case "perpendicular to the viewing ray" reduces
to the screen-space direction perpendicular to the projected viewing
direction (the in-plane tangent). This module implements that 2D
specialisation.

NOTE: this is the PyTorch pure-functional reference implementation —
deliberately slow but correct. It exists so the V6Model can run end to
end while the production CUDA kernels are written in parallel. Treat
this file as the *spec* the kernel must reproduce.
"""

from __future__ import annotations

import torch

__all__ = ["perpendicular_dilation"]


def perpendicular_dilation(
    sigma_2d: torch.Tensor,
    view_direction: torch.Tensor,
    epsilon: float = 0.5,
) -> torch.Tensor:
    """Add ``epsilon`` to the eigenvalue of ``sigma_2d`` along the axis
    perpendicular to ``view_direction``.

    Reference: AAA-Gaussians Eq. 10, 2D specialisation. The 3D form
    dilates Sigma only along directions perpendicular to the viewing ray
    so the along-ray sampling response is preserved; for a 2D-disk
    Gaussian projected to the image plane, "perpendicular to the
    projected viewing direction" is a single in-plane axis, so the
    dilation reduces to adding ``epsilon`` to the eigenvalue along that
    axis.

    The default ``epsilon = 0.5`` (half a pixel squared) matches the
    AAA-Gaussians-recommended baseline: a half-pixel worst-case
    sampling-rate variance bound.

    Construction: form the unit perpendicular ``n_perp`` to
    ``view_direction``; the rank-1 update ``epsilon * n_perp n_perp^T``
    adds ``epsilon`` to the eigenvalue along ``n_perp`` and leaves the
    other eigenvalue untouched. Returns ``sigma_2d + epsilon *
    n_perp n_perp^T``.

    Args:
        sigma_2d:        (..., 2, 2) projected screen-space covariance.
        view_direction:  (..., 2) projected viewing direction in screen
            space. Need not be unit-length; this routine normalises.
        epsilon:         dilation strength in pixel^2 added perpendicular
            to ``view_direction``. Default 0.5.

    Returns:
        (..., 2, 2) dilated covariance, broadcast over leading dims.

    Notes:
        * If ``view_direction`` has zero magnitude (degenerate, e.g. a
          Gaussian sitting on the optical axis with no projected motion),
          we fall back to dilating along the +x axis. The choice is
          arbitrary because the rasterizer caller should not be invoking
          dilation on a perfectly head-on splat in the first place.
        * bf16-safe: only matmul / outer / addition operators, all of
          which are autocast-stable.
    """
    if sigma_2d.shape[-2:] != (2, 2):
        raise ValueError(f"sigma_2d must be (..., 2, 2), got {tuple(sigma_2d.shape)}")
    if view_direction.shape[-1] != 2:
        raise ValueError(
            f"view_direction must be (..., 2), got {tuple(view_direction.shape)}"
        )

    # Normalise view direction. Use a small floor to avoid div-by-zero on
    # degenerate (head-on) inputs; the floor is below any meaningful
    # screen-space magnitude.
    norm = torch.linalg.vector_norm(view_direction, dim=-1, keepdim=True)
    safe_norm = torch.clamp(norm, min=1.0e-12)
    v = view_direction / safe_norm

    # If the view direction was effectively zero, fall back to +x.
    fallback = torch.zeros_like(v)
    fallback[..., 0] = 1.0
    v = torch.where(norm > 1.0e-12, v, fallback)

    # Perpendicular in 2D: rotate 90 deg.
    # n_perp = (-v_y, v_x)
    n_perp = torch.stack((-v[..., 1], v[..., 0]), dim=-1)  # (..., 2)

    # Outer product n_perp n_perp^T -> (..., 2, 2).
    outer = n_perp.unsqueeze(-1) * n_perp.unsqueeze(-2)
    return sigma_2d + epsilon * outer

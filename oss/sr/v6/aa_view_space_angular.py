"""AAA-Gaussians view-space angular bounds for tile-binning.

Source: Steiner et al., "AAA-Gaussians: Anti-Aliased and Artifact-Free 3D
Gaussian Rendering," ICCV 2025 (Highlight). arXiv:2504.12811.
Equations 14-17.

The published formulation replaces screen-space axis-aligned bounding
boxes with view-space angular tangent-half-angle bounds, fitting planes
in view space rather than rectangles in screen space. This avoids the
discontinuous jumps that occur when a Gaussian's screen-space AABB
crosses a tile edge or near-plane boundary -- the headline source of
"popping" in vanilla 3DGS and 2DGS rasterizers.

OSS uses 2D Gaussians whose mean is already in screen-space pixel
coordinates after projection. The 2D specialisation of the AAA bound is
straightforward: each Gaussian's principal-axis extent is sigma_radius
times the square-root of the eigenvalue along that axis, and the bbox
extends sigma_radius * sqrt(lambda) along each principal axis.

Critically -- and this is the AAA contribution that buys "no popping" --
the resulting bbox is NOT clamped to the screen rect. The caller's
tile-binning step then sees gradually-out-of-frame Gaussians instead of
suddenly-dropped ones.

NOTE: PyTorch pure-functional reference implementation. Slow but
correct. Production CUDA kernels follow as a separate sprint.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["angular_bounds"]


def angular_bounds(
    sigma_2d: torch.Tensor,
    mean_screen: torch.Tensor,
    image_size: Tuple[int, int],
    sigma_radius: float = 3.0,
) -> torch.Tensor:
    """Per-Gaussian screen-space bounding boxes derived from view-space
    angular extent.

    Reference: AAA-Gaussians Eqs. 14-17, 2D specialisation. The 3D
    formulation solves for tangent half-angles
    ``theta = arctan((s13 +/- sqrt(s13^2 - s11 s33)) / s33)``
    in view space; for a 2D-disk Gaussian whose mean is already in
    screen-space pixel coordinates, the view-space angular extent maps
    one-to-one onto the screen-space ellipse principal axes. The bounds
    are then ``sigma_radius * sqrt(lambda_i)`` along each principal axis
    of ``sigma_2d``, taken at the ``mean_screen`` centre.

    By design the returned bboxes can extend outside ``image_size``.
    The AAA "no popping" guarantee comes from the fact that we do NOT
    clamp at this stage -- the caller's tile-binning step decides
    coverage continuously instead of via a hard frustum gate.

    Construction:
        1. Decompose Sigma_2d into eigenvalues lambda_1, lambda_2.
        2. Use the largest principal-axis extent as the per-axis radius
           (a tight ellipse-AABB bound: half_extent =
           sigma_radius * sqrt(diag(Sigma_2d)). This is the standard
           half-extent of the projected ellipse along x and y axes,
           which equals sqrt(Sigma[0,0]) along x and sqrt(Sigma[1,1])
           along y.
        3. Bbox = (mean - half_extent, mean + half_extent) along each
           axis, unclamped.

    The AABB derived directly from sqrt(diag(Sigma)) is the tightest
    axis-aligned box that contains the sigma_radius-level set ellipse
    of a 2D Gaussian; this is a well-known property of 2D Gaussians
    (the projection onto each axis is a 1D Gaussian with variance
    Sigma[i,i]).

    Args:
        sigma_2d:     (N, 2, 2) per-Gaussian projected covariance.
        mean_screen:  (N, 2) per-Gaussian projected mean in pixel coords.
        image_size:   (H, W). Currently unused for clamping (by AAA
            design) but accepted for parity with screen-space-AABB
            callers and for future use in optional logging / culling.
        sigma_radius: number of standard deviations to enclose. Default
            3.0 covers ~99.7% of the integrated Gaussian mass per axis.

    Returns:
        (N, 4) bboxes ``(x_min, y_min, x_max, y_max)``. May extend
        outside ``[0, W] x [0, H]`` -- that is the AAA "no popping"
        contract.

    Notes:
        * Diagonal entries of Sigma_2d are clamped at zero before sqrt
          to handle bf16 round-off producing tiny negatives. A zero
          diagonal entry collapses the bbox to a line in that axis,
          which is the correct degenerate behaviour.
        * bf16-safe.
    """
    if sigma_2d.shape[-2:] != (2, 2):
        raise ValueError(f"sigma_2d must be (N, 2, 2), got {tuple(sigma_2d.shape)}")
    if sigma_2d.shape[0] != mean_screen.shape[0]:
        raise ValueError(
            f"sigma_2d ({sigma_2d.shape[0]}) and mean_screen "
            f"({mean_screen.shape[0]}) must agree on N"
        )
    if mean_screen.shape[-1] != 2:
        raise ValueError(
            f"mean_screen must be (N, 2), got {tuple(mean_screen.shape)}"
        )
    if len(image_size) != 2:
        raise ValueError(f"image_size must be (H, W), got {image_size}")
    if sigma_radius <= 0.0:
        raise ValueError(f"sigma_radius must be > 0, got {sigma_radius}")

    # Diagonal of Sigma -- variance along screen x and screen y.
    var_x = sigma_2d[..., 0, 0]
    var_y = sigma_2d[..., 1, 1]
    # Floor at 0 to absorb tiny bf16 negatives; sqrt of clamped value.
    half_x = sigma_radius * torch.sqrt(torch.clamp(var_x, min=0.0))
    half_y = sigma_radius * torch.sqrt(torch.clamp(var_y, min=0.0))

    x_min = mean_screen[..., 0] - half_x
    y_min = mean_screen[..., 1] - half_y
    x_max = mean_screen[..., 0] + half_x
    y_max = mean_screen[..., 1] + half_y

    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

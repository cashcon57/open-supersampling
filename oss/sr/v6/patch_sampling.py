"""v6 patch sampling: 70% importance-weighted (variance) + 30% uniform.

Per the v6 architecture memo (``docs/superpowers/experiments/
2026-05-05-v6-architecture-canonical.md``) section 6:

    "Patch sampling: 70% importance-sampled (variance-weighted) + 30% uniform"

Importance is computed from Sobel gradient magnitude integrated over a
``patch_size x patch_size`` box. Patches anchored at high-gradient regions
get sampled more often, biasing the trainer toward the hard parts of the
image (edges, texture) while still preserving flat-region quality via
the 30% uniform draw.

Public entry point:

    importance_weighted_patch_indices(image, patch_size, num_patches,
                                      importance_ratio=0.7)
        -> list[(top, left)]
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel gradient magnitude
# ---------------------------------------------------------------------------


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]]
)


def sobel_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel Sobel gradient magnitude.

    Args:
        image: (C, H, W) float tensor in any range.

    Returns:
        (1, H, W) float tensor — gradient magnitude summed across channels.
    """
    if image.dim() != 3:
        raise ValueError(f"sobel_gradient_magnitude expects (C, H, W); got {tuple(image.shape)}")
    C, H, W = image.shape
    kx = _SOBEL_X.to(dtype=image.dtype, device=image.device).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    ky = _SOBEL_Y.to(dtype=image.dtype, device=image.device).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    gx = F.conv2d(image.unsqueeze(0), kx, padding=1, groups=C)
    gy = F.conv2d(image.unsqueeze(0), ky, padding=1, groups=C)
    mag = (gx.pow(2) + gy.pow(2)).clamp(min=1e-12).sqrt()
    # Reduce across channels.
    mag = mag.sum(dim=1, keepdim=False)  # (1, H, W)
    return mag


# ---------------------------------------------------------------------------
# Patch index sampling
# ---------------------------------------------------------------------------


def importance_weighted_patch_indices(
    image: torch.Tensor,
    patch_size: int,
    num_patches: int,
    importance_ratio: float = 0.7,
    *,
    generator: torch.Generator | None = None,
) -> List[Tuple[int, int]]:
    """Return ``num_patches`` patch top-left coordinates.

    ``importance_ratio`` of them are drawn from a distribution
    proportional to local Sobel-gradient magnitude integrated over a
    ``patch_size x patch_size`` box; the remainder are drawn uniformly.

    Sampling is without replacement *within each pool* (importance and
    uniform are sampled independently). If the importance distribution
    has fewer non-zero anchors than the requested importance count, the
    shortfall is moved into the uniform pool.

    Args:
        image: (C, H, W) source image. Used only to compute the importance
            distribution; the returned coordinates are valid against any
            tensor of the same spatial shape.
        patch_size: side length of the square patch (positive).
        num_patches: total patches to return (>= 0).
        importance_ratio: fraction in [0, 1].
        generator: optional torch.Generator for deterministic sampling.

    Returns:
        List of ``(top, left)`` tuples, length ``num_patches``. Every
        coordinate satisfies ``0 <= top <= H - patch_size`` and similarly
        for ``left``.
    """
    if image.dim() != 3:
        raise ValueError(
            f"importance_weighted_patch_indices expects (C, H, W); got {tuple(image.shape)}"
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive; got {patch_size}")
    if num_patches < 0:
        raise ValueError(f"num_patches must be non-negative; got {num_patches}")
    if not (0.0 <= importance_ratio <= 1.0):
        raise ValueError(
            f"importance_ratio must be in [0, 1]; got {importance_ratio}"
        )

    _, H, W = image.shape
    if H < patch_size or W < patch_size:
        raise ValueError(
            f"image spatial size ({H}, {W}) is smaller than patch_size {patch_size}; "
            "cannot extract any patches"
        )

    if num_patches == 0:
        return []

    # Number of valid anchor positions.
    anchor_h = H - patch_size + 1
    anchor_w = W - patch_size + 1
    n_anchors = anchor_h * anchor_w

    n_importance = int(round(num_patches * importance_ratio))
    n_uniform = num_patches - n_importance

    importance_coords: List[Tuple[int, int]] = []
    if n_importance > 0:
        # Compute the integrated-gradient importance map over patch_size boxes.
        mag = sobel_gradient_magnitude(image)  # (1, H, W) — note: returns shape (1,H,W)
        # Use avg_pool2d on the magnitude with kernel = patch_size, stride 1.
        mag_b = mag.unsqueeze(0)  # (1, 1, H, W) — wait sobel returns (1,H,W) so unsqueeze once
        # Re-shape: sobel_gradient_magnitude returns (1, H, W)? we built (1,H,W)
        # actually we built (1,H,W) -> unsqueeze to (1,1,H,W).
        if mag_b.dim() == 3:
            mag_b = mag_b.unsqueeze(0)
        integrated = F.avg_pool2d(
            mag_b, kernel_size=patch_size, stride=1, padding=0
        ).squeeze(0).squeeze(0)  # (anchor_h, anchor_w)

        weights = integrated.flatten()
        weights = weights.clamp(min=0.0)
        total = weights.sum()
        if total <= 0:
            # Degenerate (image is uniform) — fall back to uniform sampling
            # for the importance pool too.
            n_uniform += n_importance
            n_importance = 0
        else:
            probs = weights / total
            # Multinomial without replacement. Cap to available anchors.
            k = min(n_importance, n_anchors)
            samples = torch.multinomial(
                probs, num_samples=k, replacement=False, generator=generator,
            )
            for s in samples.tolist():
                top = s // anchor_w
                left = s % anchor_w
                importance_coords.append((int(top), int(left)))
            # Move any shortfall into the uniform pool.
            shortfall = n_importance - k
            n_uniform += shortfall
            n_importance = k

    uniform_coords: List[Tuple[int, int]] = []
    if n_uniform > 0:
        k_u = min(n_uniform, n_anchors)
        # Uniform without replacement via randperm; cheap for typical anchor
        # counts. For very large maps this could use multinomial w/ uniform
        # weights, but anchor count for our patch sizes is bounded.
        if generator is not None:
            perm = torch.randperm(n_anchors, generator=generator)[:k_u]
        else:
            perm = torch.randperm(n_anchors)[:k_u]
        for s in perm.tolist():
            top = s // anchor_w
            left = s % anchor_w
            uniform_coords.append((int(top), int(left)))
        shortfall = n_uniform - k_u
        if shortfall > 0:
            # Need more patches than there are anchor positions: sample with
            # replacement to make up the requested total. Prefer importance
            # weights when available, else uniform.
            if n_importance > 0 and importance_coords:
                pad = torch.multinomial(
                    weights / weights.sum(), num_samples=shortfall,
                    replacement=True, generator=generator,
                )
                for s in pad.tolist():
                    top = s // anchor_w
                    left = s % anchor_w
                    uniform_coords.append((int(top), int(left)))
            else:
                if generator is not None:
                    extra = torch.randint(0, n_anchors, (shortfall,), generator=generator)
                else:
                    extra = torch.randint(0, n_anchors, (shortfall,))
                for s in extra.tolist():
                    top = s // anchor_w
                    left = s % anchor_w
                    uniform_coords.append((int(top), int(left)))

    coords = importance_coords + uniform_coords
    # Trim/pad to exactly num_patches (defensive — should already match).
    if len(coords) > num_patches:
        coords = coords[:num_patches]
    return coords


__all__ = [
    "sobel_gradient_magnitude",
    "importance_weighted_patch_indices",
]

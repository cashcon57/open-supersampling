"""Opacity-threshold + count-cap pruning for the v5 Gaussian temporal track.

This operation is **non-differentiable** by design: it mutates the boolean
``alive`` mask of a :class:`GaussianField` based on opacity comparisons and
ranking. It must be invoked **after** the loss / backward pass has completed
on the rendered output for the current frame — running it inside the autograd
graph would either be a no-op (no gradient flows through bool indexing) or, in
the case of count-cap eviction, silently drop gradient paths.

Two operations, applied in order:

1. **Opacity threshold:** any Gaussian whose ``opacity < opacity_threshold``
   has its ``alive`` flag cleared.
2. **Count cap:** if the surviving alive count exceeds ``max_count``, the
   ``count_alive() - max_count`` lowest-opacity alive Gaussians are evicted
   (``alive`` cleared) so that ``count_alive() <= max_count``.
"""
from __future__ import annotations

import torch

from oss.sr.gaussian_temporal.gaussian_field import GaussianField


@torch.no_grad()
def prune(
    field: GaussianField, opacity_threshold: float, max_count: int
) -> GaussianField:
    """Drop low-opacity Gaussians and enforce a hard ``max_count`` cap.

    Non-differentiable; call **after** the per-frame loss/backward pass.

    Args:
        field: input :class:`GaussianField` (not mutated; a clone is returned).
        opacity_threshold: alive Gaussians with ``opacity < opacity_threshold``
            are marked dead.
        max_count: hard cap on ``count_alive()``. If exceeded after the
            opacity threshold pass, the lowest-opacity alive Gaussians are
            evicted until ``count_alive() <= max_count``.

    Returns:
        A new :class:`GaussianField` with the updated ``alive`` mask.
    """
    out = field.clone()
    low = (out.opacity < opacity_threshold) & out.alive
    out.alive[low] = False
    n_alive = int(out.alive.sum().item())
    if n_alive > max_count:
        live_idx = out.alive.nonzero(as_tuple=True)[0]
        ranked = live_idx[torch.argsort(out.opacity[live_idx])]
        evict = ranked[: n_alive - max_count]
        out.alive[evict] = False
    return out


__all__ = ["prune"]

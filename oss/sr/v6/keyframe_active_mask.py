"""4DGS-1K key-frame active-Gaussian mask cache.

Reference: Yuan et al., 4DGS-1K (arXiv:2503.16422). Active-Gaussian sets
overlap ~85-95% between adjacent frames in streaming workloads, so
recomputing per-frame visibility is wasteful. We compute a binary
visibility mask on every K-th frame and reuse the nearest keyframe's
mask on intermediates.

Activeness on a keyframe = "Gaussian's projected screen-space 3-sigma
bbox overlaps the viewport." The 3-sigma extent is the same envelope
the rasterizer uses for tile assignment (see
``oss/gaussian/renderer/rasterizer.py``); we recompute it here to keep
the cache decoupled from the rasterizer's tiling internals.

Not an ``nn.Module``: this is a stateful cache, not a learned layer.
"""
from __future__ import annotations

import torch


def _project_centers(positions: torch.Tensor, view_matrix: torch.Tensor) -> torch.Tensor:
    """Project Gaussian centers through ``view_matrix`` to screen space.

    The v6 pipeline uses 2D pixel-space Gaussian centers (see
    ``PersistentCanvas``); the view matrix is therefore an affine 2D->2D
    transform. We support three common shapes:

    * ``(2, 3)`` affine: ``[R | t]``
    * ``(3, 3)`` homogeneous affine: last row implicit ``[0, 0, 1]``
    * ``(2, 2)`` rotation/scale only (zero translation)

    Returns ``(N, 2)`` projected pixel coordinates.
    """
    if positions.dim() != 2 or positions.shape[-1] != 2:
        raise ValueError(f"positions must be (N, 2); got {tuple(positions.shape)}")
    vm = view_matrix
    if vm.dim() != 2:
        raise ValueError(f"view_matrix must be 2D; got {tuple(vm.shape)}")

    if vm.shape == (2, 2):
        return positions @ vm.transpose(0, 1)
    if vm.shape == (2, 3):
        rot = vm[:, :2]
        trans = vm[:, 2]
        return positions @ rot.transpose(0, 1) + trans
    if vm.shape == (3, 3):
        rot = vm[:2, :2]
        trans = vm[:2, 2]
        return positions @ rot.transpose(0, 1) + trans
    raise ValueError(
        f"view_matrix must be (2, 2), (2, 3), or (3, 3); got {tuple(vm.shape)}"
    )


def _three_sigma_radius(scales: torch.Tensor) -> torch.Tensor:
    """3-sigma envelope radius per Gaussian.

    For axis-aligned scales we take the max axis. ``scales`` may be
    ``(N, 2)`` (per-axis) or ``(N,)`` (isotropic). We deliberately ignore
    rotation here — the bounding-box overlap test is conservative under
    rotation (axis-aligned 3-sigma circumscribes the rotated ellipse).
    """
    if scales.dim() == 1:
        return 3.0 * scales.abs()
    if scales.dim() == 2 and scales.shape[-1] == 2:
        return 3.0 * scales.abs().amax(dim=-1)
    raise ValueError(f"scales must be (N,) or (N, 2); got {tuple(scales.shape)}")


def _compute_active_mask(
    canvas,
    view_matrix: torch.Tensor | None,
    viewport_hw: tuple[int, int] | None,
) -> torch.Tensor:
    """Binary mask of Gaussians whose projected 3-sigma bbox hits the viewport."""
    if not hasattr(canvas, "positions") or not hasattr(canvas, "scales"):
        raise AttributeError(
            "canvas must expose 'positions' and 'scales' for active-mask "
            "computation (duck-typed Gaussian field interface)"
        )
    positions = canvas.positions
    scales = canvas.scales

    n_total = positions.shape[0]
    n_live = int(getattr(canvas, "count", n_total))
    n_live = max(0, min(n_live, n_total))
    if n_live == 0:
        return torch.zeros(n_total, dtype=torch.bool, device=positions.device)

    # Restrict to the live prefix; pad the inactive tail with False below.
    live_pos = positions[:n_live]
    live_scales = scales[:n_live]

    if view_matrix is None:
        proj = live_pos
    else:
        proj = _project_centers(
            live_pos,
            view_matrix.to(live_pos.device).to(live_pos.dtype),
        )
    radius = _three_sigma_radius(live_scales).to(proj.dtype)

    if viewport_hw is None:
        # Fall back to the canvas's declared output shape if it has one.
        if hasattr(canvas, "output_hw"):
            viewport_hw = tuple(canvas.output_hw)
        else:
            raise ValueError(
                "viewport_hw not provided and canvas has no .output_hw fallback"
            )
    h, w = int(viewport_hw[0]), int(viewport_hw[1])

    x = proj[:, 0]
    y = proj[:, 1]
    # AABB-overlap-with-rect test: any part of [x-r, x+r] x [y-r, y+r] inside [0,W) x [0,H).
    overlaps = (x + radius >= 0) & (x - radius < w) & (y + radius >= 0) & (y - radius < h)

    out = torch.zeros(n_total, dtype=torch.bool, device=positions.device)
    out[:n_live] = overlaps
    return out


class KeyframeActiveMaskCache:
    """Per-K-frame visibility cache for 4DGS-1K-style streaming pruning.

    Usage:

        cache = KeyframeActiveMaskCache(keyframe_interval=10)
        for frame_idx, view in enumerate(frames):
            mask = cache.get_mask(frame_idx, canvas, view, viewport_hw=(H, W))
            ...

    The mask is recomputed when ``frame_index % keyframe_interval == 0``
    OR when the cache is empty (first call). Otherwise the most recently
    computed keyframe mask is returned. ``reset()`` clears the cache so
    the next ``get_mask`` recomputes regardless of frame index — call this
    on canvas-rebuild boundaries.
    """

    def __init__(self, keyframe_interval: int = 10) -> None:
        if keyframe_interval <= 0:
            raise ValueError(
                f"keyframe_interval must be positive; got {keyframe_interval}"
            )
        self.keyframe_interval = int(keyframe_interval)
        self._cached_mask: torch.Tensor | None = None
        self._cached_keyframe: int | None = None

    def reset(self) -> None:
        """Clear the cached mask. Next ``get_mask`` recomputes."""
        self._cached_mask = None
        self._cached_keyframe = None

    def get_mask(
        self,
        frame_index: int,
        canvas,
        view_matrix: torch.Tensor | None,
        viewport_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return ``(N,)`` boolean active mask for ``frame_index``.

        Args:
            frame_index: monotonically increasing frame counter. Used only
                to decide whether this frame is a keyframe.
            canvas: duck-typed Gaussian field (must expose ``positions``,
                ``scales``; may expose ``count``, ``output_hw``).
            view_matrix: 2D affine transform from canvas-space to
                screen-space (see ``_project_centers``). ``None`` is
                treated as identity.
            viewport_hw: ``(H, W)``. Defaults to ``canvas.output_hw`` if
                present.

        Returns:
            ``(N,)`` bool tensor. ``True`` = active (rasterize).
        """
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative; got {frame_index}")

        is_keyframe = (frame_index % self.keyframe_interval == 0)
        n_total = canvas.positions.shape[0]
        cache_invalid = (
            self._cached_mask is None
            or self._cached_mask.shape[0] != n_total
            or self._cached_mask.device != canvas.positions.device
        )

        if is_keyframe or cache_invalid:
            self._cached_mask = _compute_active_mask(canvas, view_matrix, viewport_hw)
            self._cached_keyframe = frame_index

        # Hand out a clone so callers can't poison the cache by mutating in place.
        return self._cached_mask.clone()


__all__ = ["KeyframeActiveMaskCache"]

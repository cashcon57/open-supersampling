"""Motion-vector warp for the persistent Gaussian canvas — Sprint 5 / T5.2.

Each frame, every Gaussian's centre samples the per-pixel motion vector
field at its (sub-pixel) position via bilinear interpolation, then shifts
by that sampled vector. Gaussians whose post-warp centre falls outside
the frame are flagged via the ``in_frame`` mask so the prune step can
retire them.

Pure PyTorch; runs unchanged on CPU and CUDA. No autograd is required —
the canvas is state, not a parameter — so we do everything under
``no_grad`` semantics by working on detached tensors.

Design doc: ``docs/superpowers/gaussian-canvas-design.md`` §1, §3.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from oss.gaussian.canvas.canvas import PersistentCanvas


def warp_positions(
    xy: torch.Tensor,
    motion: torch.Tensor,
    hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a motion-vector field to a set of Gaussian positions.

    Args:
        xy:     ``(N, 2)`` sub-pixel positions in pixel-space
                ``[0, W) × [0, H)``. Order is (x, y).
        motion: ``(2, H, W)`` motion vectors. Channel 0 is dx, channel 1
                is dy. Same units as ``xy`` (pixels per frame).
        hw:     ``(H, W)`` of the frame the motion field describes.

    Returns:
        ``(new_xy, in_frame)``:
        - ``new_xy``: ``(N, 2)`` positions after warp.
        - ``in_frame``: ``(N,)`` bool. ``True`` iff the new position lies
          inside ``[0, W) × [0, H)``. The prune step retires the rest.

    The bilinear sample uses ``F.grid_sample`` with zero-padding outside
    the frame; out-of-frame Gaussians therefore receive zero motion and
    sit where they were warped to from the prior frame's flag — fine,
    because ``in_frame`` is the authoritative liveness signal.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2); got {tuple(xy.shape)}")
    if motion.ndim != 3 or motion.shape[0] != 2:
        raise ValueError(f"motion must be (2, H, W); got {tuple(motion.shape)}")
    h, w = hw
    if motion.shape[1] != h or motion.shape[2] != w:
        raise ValueError(
            f"motion spatial dims {tuple(motion.shape[1:])} disagree with hw={hw}"
        )

    n = xy.shape[0]
    if n == 0:
        return xy.clone(), torch.zeros((0,), dtype=torch.bool, device=xy.device)

    # Map xy → normalised grid coords in [-1, 1].
    # grid_sample expects (x_norm, y_norm) per sample point with x along W.
    # Pixel centre at integer i has normalised coord (2i+1)/N - 1 (align_corners=False).
    x_norm = (xy[:, 0] / w) * 2.0 - 1.0  # (N,)
    y_norm = (xy[:, 1] / h) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, n, 1, 2)  # (1, N, 1, 2)

    # motion shape (2, H, W) → (1, 2, H, W)
    sampled = F.grid_sample(
        motion.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )  # (1, 2, N, 1)
    dxdy = sampled.view(2, n).t().contiguous()  # (N, 2)

    new_xy = xy + dxdy
    in_frame = (
        (new_xy[:, 0] >= 0)
        & (new_xy[:, 0] < float(w))
        & (new_xy[:, 1] >= 0)
        & (new_xy[:, 1] < float(h))
    )
    return new_xy, in_frame


def warp_canvas(
    canvas: "PersistentCanvas",
    motion: torch.Tensor,
    alpha: float = 1.0,
) -> "PersistentCanvas":
    """Return a new ``PersistentCanvas`` with positions shifted by
    ``motion × alpha``. Covariance, rotation, and colour are reused
    unchanged (per design spec §3.2 — covariance frozen).

    This is the public entry point Sprint 6 (frame extrapolation) builds
    against. ``alpha < 1`` produces a fractional shift used for predicted
    frames at ``t + α``.

    Args:
        canvas: source canvas at frame t.
        motion: ``(2, H, W)`` motion field at the canvas's output
                resolution. Channel 0 is dx, channel 1 is dy.
        alpha:  scalar in ``[0, 1]``. ``0`` → no shift; ``1`` → full
                t-1 → t shift.

    Returns:
        A shallow-cloned ``PersistentCanvas`` with shifted positions and
        an updated ``alive`` flag for any Gaussian warped out of frame.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    new_canvas = copy.copy(canvas)
    # Clone the per-frame mutating state so the source canvas stays intact.
    new_canvas.positions = canvas.positions.clone()
    new_canvas.alive = canvas.alive.clone()
    new_canvas.age = canvas.age.clone()
    new_canvas.error = canvas.error.clone()
    # Geometry/colour tensors are shared (frozen per the design doc) — no
    # per-frame mutation, so a shared reference is correct.

    new_xy, in_frame = warp_positions(
        new_canvas.positions, motion * float(alpha), new_canvas.output_hw
    )
    new_canvas.positions = new_xy
    # Drop out-of-frame Gaussians from the alive set so the snapshot stays
    # well-formed for the renderer.
    new_canvas.alive = new_canvas.alive & in_frame
    return new_canvas


__all__ = ["warp_positions", "warp_canvas"]

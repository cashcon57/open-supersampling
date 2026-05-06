"""4DGS-1K Spatial-Temporal Variation Score pruning.

Reference: Yuan et al., ``1000+ FPS 4D Gaussian Splatting``, NeurIPS 2025
(arXiv:2503.16422). Per-Gaussian importance score:

    S_i = SS_i * TS_i

* ``SS_i`` (Spatial Score): aggregate ``alpha_i * T_i`` contribution across
  every observed pixel of every observed frame. Approximates the Gaussian's
  visible footprint integrated over time.
* ``TS_i`` (Temporal Score): number of frames the Gaussian was alive
  (in the active mask). Long-lived Gaussians outscore transient ones.

Globally rank Gaussians by ``S_i`` and prune the bottom fraction
(default 70%, configurable per 4DGS-1K's 60-80% sweet spot).

The canvas argument is duck-typed to keep this module decoupled from the
concrete ``PersistentCanvas`` impl in ``oss/gaussian/canvas/canvas.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class STVScoreState:
    """Running aggregate of spatial contribution and lifespan per Gaussian.

    Attributes:
        spatial_accumulator: ``(N,)`` running sum of per-Gaussian
            ``sum_p (alpha * T)`` contributions across observed frames.
            Float dtype matching the rasterizer's working precision.
        lifespan_count: ``(N,)`` int64 count of frames the Gaussian was
            in the active mask.
        frames_observed: total frames folded into the state. Used by
            callers to gate "do we have enough samples to prune yet".
    """

    spatial_accumulator: torch.Tensor
    lifespan_count: torch.Tensor
    frames_observed: int


def init_st_score_state(
    n: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> STVScoreState:
    """Allocate a zero-initialized state for a canvas of ``n`` Gaussians."""
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    return STVScoreState(
        spatial_accumulator=torch.zeros(n, device=device, dtype=dtype),
        lifespan_count=torch.zeros(n, device=device, dtype=torch.int64),
        frames_observed=0,
    )


def update_st_score(
    state: STVScoreState,
    alpha_per_pixel: torch.Tensor,
    transmittance_per_pixel: torch.Tensor,
    is_active: torch.Tensor,
) -> STVScoreState:
    """Fold one frame of (alpha, T, active) observations into the state.

    Args:
        state: prior state. Shape determines ``N``; this function does NOT
            grow the buffer — caller must rebuild on canvas resize.
        alpha_per_pixel: ``(N, P)`` per-Gaussian per-pixel opacity
            contribution this frame. ``P`` is the rasterizer's flattened
            pixel count (or any consistent reduction axis).
        transmittance_per_pixel: ``(N, P)`` per-Gaussian per-pixel
            transmittance ``T_i`` along the splat-sort order.
        is_active: ``(N,)`` boolean mask of Gaussians counted as alive
            this frame (typically the keyframe active mask).

    Returns:
        Updated state (in-place; same object returned for convenience).
    """
    n = state.spatial_accumulator.shape[0]
    if alpha_per_pixel.dim() != 2 or alpha_per_pixel.shape[0] != n:
        raise ValueError(
            f"alpha_per_pixel must be (N, P) with N={n}; got {tuple(alpha_per_pixel.shape)}"
        )
    if transmittance_per_pixel.shape != alpha_per_pixel.shape:
        raise ValueError(
            f"transmittance_per_pixel shape {tuple(transmittance_per_pixel.shape)} "
            f"!= alpha_per_pixel {tuple(alpha_per_pixel.shape)}"
        )
    if is_active.shape != (n,):
        raise ValueError(f"is_active must be (N,) with N={n}; got {tuple(is_active.shape)}")

    # Per-Gaussian visible-footprint contribution this frame.
    # bf16-safe: promote to float32 for the accumulation, cast back.
    work_dtype = state.spatial_accumulator.dtype
    contrib = (alpha_per_pixel.to(work_dtype) * transmittance_per_pixel.to(work_dtype)).sum(dim=1)

    state.spatial_accumulator = state.spatial_accumulator + contrib
    state.lifespan_count = state.lifespan_count + is_active.to(torch.int64)
    state.frames_observed += 1
    return state


def st_variation_score(state: STVScoreState) -> torch.Tensor:
    """Compute ``S_i = SS_i * TS_i`` from the running state.

    Returns ``(N,)`` float32 score. Empty state returns an empty tensor.
    """
    ss = state.spatial_accumulator
    ts = state.lifespan_count.to(ss.dtype)
    return ss * ts


def _gather_canvas_attrs(canvas) -> dict[str, torch.Tensor]:
    """Pull the duck-typed canvas attribute set into a dict.

    The interface matches the NeurIPS 2025 4DGS-1K formulation: positions,
    scales, rotations, opacities, colors. ``count`` is the live-Gaussian
    count.
    """
    required = ("positions", "scales", "rotations", "opacities", "colors")
    out: dict[str, torch.Tensor] = {}
    for name in required:
        if not hasattr(canvas, name):
            raise AttributeError(
                f"canvas missing required attribute {name!r}; "
                f"expected duck-typed surface (positions/scales/rotations/opacities/colors/count)"
            )
        out[name] = getattr(canvas, name)
    return out


def prune_by_st_score(
    canvas,
    state: STVScoreState,
    prune_fraction: float = 0.7,
):
    """Prune the bottom ``prune_fraction`` of Gaussians by S-T variation score.

    Args:
        canvas: duck-typed Gaussian field with ``.positions``, ``.scales``,
            ``.rotations``, ``.opacities``, ``.colors``, ``.count``. Mutated
            in place: each tensor attr is reassigned to the kept subset and
            ``count`` is updated.
        state: STVScoreState with N == canvas.count.
        prune_fraction: in ``[0, 1)``. ``0`` keeps everything; ``1`` would
            prune everything (rejected).

    Returns:
        ``(canvas, new_state)`` — the same canvas object (mutated) and a
        fresh state reduced to the surviving Gaussians.
    """
    if not (0.0 <= prune_fraction < 1.0):
        raise ValueError(
            f"prune_fraction must be in [0, 1); got {prune_fraction}"
        )

    attrs = _gather_canvas_attrs(canvas)
    n = int(getattr(canvas, "count"))

    if n == 0 or prune_fraction == 0.0:
        # Nothing to do; hand back an unchanged state.
        return canvas, state

    if state.spatial_accumulator.shape[0] != n:
        raise ValueError(
            f"state size {state.spatial_accumulator.shape[0]} != canvas.count {n}"
        )

    score = st_variation_score(state)
    n_keep = max(1, int(round(n * (1.0 - prune_fraction))))

    # ``topk`` on ``score[:n]`` — only the live prefix is scored. If the
    # canvas stores trailing dead slots, callers should slice before calling.
    _, keep_idx = torch.topk(score, k=n_keep, largest=True, sorted=False)
    # Stable order keeps the relative layout for cheaper downstream re-warps.
    keep_idx, _ = torch.sort(keep_idx)

    for name, tensor in attrs.items():
        setattr(canvas, name, tensor[keep_idx].contiguous())
    setattr(canvas, "count", n_keep)

    new_state = STVScoreState(
        spatial_accumulator=state.spatial_accumulator[keep_idx].contiguous(),
        lifespan_count=state.lifespan_count[keep_idx].contiguous(),
        frames_observed=state.frames_observed,
    )
    return canvas, new_state


__all__ = [
    "STVScoreState",
    "init_st_score_state",
    "update_st_score",
    "st_variation_score",
    "prune_by_st_score",
]

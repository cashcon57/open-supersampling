"""Alpha schedules for frame-extrapolation FPS targets.

When a base render produces frames at `source_fps` and the display target
is `target_fps`, the extrapolator must emit `target_fps / source_fps - 1`
intermediate frames between every pair of real frames. Each intermediate
frame is rendered with a different alpha in (0, 1).

Examples (source = 60 fps):

    target 90 fps  → 1 intermediate per real frame at alpha = 0.5
    target 120 fps → 1 intermediate per real frame at alpha = 0.5
                     (display real → alpha=0 → real-1 sequence; the warp
                      magnitude that DLSS-FG-equivalent ratio produces is
                      0.5 because we double frame count)
    target 144 fps → 1.4 intermediates per real frame on average; the
                     scheduler emits `[0.417, 0.833]` over a 5-frame
                     cadence so the long-run rate matches.

The scheduler returns the alphas in **display order** for one cadence
period; callers loop the returned list. Real frames (alpha=0) are
implicit — the scheduler emits only the synthesized alphas.

Quality note (see docs/superpowers/gaussian-frame-extrapolation.md):
high alpha values approach the next-frame prediction and so accumulate
the most non-linear-motion error. Where multiple intermediates are
needed, the scheduler distributes them uniformly to bound worst-case
warp distance per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import List


@dataclass(frozen=True)
class AlphaSchedule:
    """A periodic alpha schedule for an integer cadence ratio.

    Attributes:
        source_fps: frames per second produced by the real renderer.
        target_fps: total frames per second emitted to the display.
        alphas: list of alpha values for each synthesized intermediate
            frame in one cadence period, in display order.
        period_frames: how many displayed frames make up one full period
            (== len(alphas) + number of real frames in the period).
    """

    source_fps: int
    target_fps: int
    alphas: List[float]
    period_frames: int

    @property
    def intermediates_per_period(self) -> int:
        return len(self.alphas)


def schedule_for(source_fps: int, target_fps: int) -> AlphaSchedule:
    """Return the alpha schedule for `source_fps` → `target_fps`.

    Requires target_fps > source_fps and target_fps % source_fps' gcd > 0.
    Produces a uniform distribution of synthesized frames between each
    real frame pair, expressed as the smallest integer cadence period.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"fps must be positive; got source={source_fps}, target={target_fps}")
    if target_fps < source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) must be >= source_fps ({source_fps}); "
            "extrapolation only inserts frames, never drops them."
        )
    if target_fps == source_fps:
        return AlphaSchedule(source_fps, target_fps, alphas=[], period_frames=1)

    # Reduce to coprime cadence: every `g` real frames, emit `target_fps/g`
    # displayed frames total. The synthesized count per period is the
    # difference.
    g = gcd(source_fps, target_fps)
    real_per_period = source_fps // g
    displayed_per_period = target_fps // g
    intermediates = displayed_per_period - real_per_period
    if intermediates <= 0:  # defensive — covered by the equality branch above
        return AlphaSchedule(source_fps, target_fps, alphas=[], period_frames=real_per_period)

    # Distribute intermediate alphas uniformly across the displayed period.
    # Real frames sit at displayed positions 0, displayed_per_period/real_per_period,
    # 2*..., etc. Synthesized frames fill the remaining slots; each slot's alpha
    # is its fractional offset from the most recent real frame, normalised by
    # the gap to the next real frame.
    alphas: List[float] = []
    real_positions = {
        i * displayed_per_period // real_per_period: i for i in range(real_per_period)
    }
    last_real_idx = 0
    next_real_pos = displayed_per_period // real_per_period
    for displayed_idx in range(displayed_per_period):
        if displayed_idx in real_positions:
            last_real_idx = real_positions[displayed_idx]
            # next real position relative to this real frame
            next_real = (last_real_idx + 1) * displayed_per_period // real_per_period
            next_real_pos = next_real
            continue
        gap = next_real_pos - (last_real_idx * displayed_per_period // real_per_period)
        offset = displayed_idx - (last_real_idx * displayed_per_period // real_per_period)
        alphas.append(offset / gap)

    assert len(alphas) == intermediates, (
        f"scheduler bug: expected {intermediates} intermediates, got {len(alphas)}"
    )
    return AlphaSchedule(
        source_fps=source_fps,
        target_fps=target_fps,
        alphas=alphas,
        period_frames=displayed_per_period,
    )


# Convenience presets — what the master plan calls out (60 → {90, 120, 144}).
def preset_60_to_90() -> AlphaSchedule:
    return schedule_for(60, 90)


def preset_60_to_120() -> AlphaSchedule:
    return schedule_for(60, 120)


def preset_60_to_144() -> AlphaSchedule:
    return schedule_for(60, 144)


__all__ = [
    "AlphaSchedule",
    "schedule_for",
    "preset_60_to_90",
    "preset_60_to_120",
    "preset_60_to_144",
]

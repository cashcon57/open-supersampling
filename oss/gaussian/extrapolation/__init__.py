"""OSS-Gaussian frame extrapolation (Sprint 6).

Public API:

- `FrameExtrapolator` — renders the persistent canvas at fractional time
   positions ``t-1 + alpha`` by reusing Sprint 5's motion-warp.
- `AlphaSchedule` — descriptor for a 60→{90,120,144} fps cadence.
- `schedule_for(source_fps, target_fps)` — build a schedule for any pair.
- `preset_60_to_{90,120,144}()` — preset schedules from the master plan.

See `docs/superpowers/plans/2026-05-01-gaussian-sprint-6-plan.md` for the
sprint plan and `docs/superpowers/gaussian-frame-extrapolation.md` for the
design rationale (vs DLSS Frame Generation, failure modes, latency budget).
"""

from oss.gaussian.extrapolation.alpha_scheduler import (
    AlphaSchedule,
    preset_60_to_90,
    preset_60_to_120,
    preset_60_to_144,
    schedule_for,
)
from oss.gaussian.extrapolation.extrapolator import FrameExtrapolator, WarpFn

__all__ = [
    "FrameExtrapolator",
    "WarpFn",
    "AlphaSchedule",
    "schedule_for",
    "preset_60_to_90",
    "preset_60_to_120",
    "preset_60_to_144",
]

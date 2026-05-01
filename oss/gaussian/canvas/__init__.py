"""OSS-Gaussian persistent canvas (Sprint 5).

GPU-resident persistent buffer of N Gaussians warped each frame by
motion vectors, with per-tile MSE driving prune+spawn. Pure PyTorch v1;
runs on CPU and CUDA from one code path.

Public API:

- ``PersistentCanvas``               — the canvas itself (state + lifecycle).
- ``CanvasStats``                    — per-frame diagnostic snapshot.
- ``warp_canvas``                    — Sprint 6 contract: returns a new
                                       canvas with positions shifted by
                                       ``motion × alpha`` (covariance frozen).
- ``warp_positions``                 — low-level bilinear motion-vector warp
                                       used by ``warp_canvas`` and the canvas
                                       update loop.
- ``per_tile_mse``                   — render-vs-LR per-tile error map.
- ``gaussians_error_from_tiles``     — per-Gaussian error from tile map.
- ``PrunePolicy``                    — tuneable prune-rule thresholds.
- ``select_for_pruning``             — pure prune-selection function.
- ``select_spawn_tiles``             — top-error tile picker.
- ``apply_prune_spawn``              — in-place state mutation.

Sprint 6 (frame extrapolation) imports ``PersistentCanvas`` and
``warp_canvas``; both are stable here.

Spec: ``docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md``
Plan: ``docs/superpowers/plans/2026-05-01-gaussian-sprint-5-plan.md``
Design notes: ``docs/superpowers/gaussian-canvas-design.md``
"""

from oss.gaussian.canvas.canvas import CanvasStats, PersistentCanvas
from oss.gaussian.canvas.error_detection import (
    gaussians_error_from_tiles,
    per_tile_mse,
)
from oss.gaussian.canvas.prune_spawn import (
    PrunePolicy,
    apply_prune_spawn,
    select_for_pruning,
    select_spawn_tiles,
)
from oss.gaussian.canvas.warp import warp_canvas, warp_positions

__all__ = [
    "PersistentCanvas",
    "CanvasStats",
    "warp_canvas",
    "warp_positions",
    "per_tile_mse",
    "gaussians_error_from_tiles",
    "PrunePolicy",
    "select_for_pruning",
    "select_spawn_tiles",
    "apply_prune_spawn",
]

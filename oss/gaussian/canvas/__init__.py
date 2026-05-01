"""OSS-Gaussian persistent canvas (Sprint 5).

Sprint 5 produces the implementations of `PersistentCanvas` and `warp_canvas`
in this package. Sprint 6 (frame extrapolation) reuses the same canvas and
warp infrastructure unchanged — the only difference is the alpha multiplier
on the motion magnitude.

This file declares the **public API contract** that Sprint 6 builds against.
Sprint 5 may extend the surface, but must not reduce or rename the symbols
listed here without coordinating with Sprint 6.

API contract
------------

    from oss.gaussian.canvas import PersistentCanvas, warp_canvas

    canvas: PersistentCanvas         # mutable container of N Gaussians
    canvas.gaussians: GaussianBatch  # current snapshot at frame t
    canvas.render(output_hw) -> (F, H, W)   # convenience render through the
                                            # Rasterizer at native resolution

    warped: PersistentCanvas = warp_canvas(canvas, motion, alpha=1.0)
        # motion: (2, H, W) per-pixel motion field at LR resolution.
        #         Convention: motion[:, y, x] = (dx, dy) describing the
        #         displacement that a feature at pixel (x, y) at frame t-1
        #         travels to reach (x+dx, y+dy) at frame t. Scale: pixels
        #         in the canvas's native coordinate frame (same as
        #         GaussianBatch.xy).
        # alpha:  scalar in [0, 1]. 0 → no shift (canvas unchanged).
        #         1 → full t-1→t shift. 0 < alpha < 1 → fractional shift,
        #         used by Sprint 6 frame extrapolation.
        # Returns a new PersistentCanvas with positions shifted; covariance,
        # rotation, and color are reused unchanged (per design spec §3.2 —
        # "covariance frozen", network handles deltas elsewhere).

Sprint 5 status
---------------

At the time Sprint 6 was scaffolded, this package was still in flight.
The frame-extrapolation code path defends against an absent implementation
by importing lazily and raising a descriptive error explaining that
Sprint 5 must land before extrapolation can run. See
`oss.gaussian.extrapolation.extrapolator.FrameExtrapolator` for the lazy
import block.

Public re-exports below stay commented out until Sprint 5 wires the
real symbols. Sprint 5's PR should uncomment them and add `__all__`.
"""

# TODO(sprint-5): Uncomment once oss/gaussian/canvas/{canvas.py,warp.py} land.
# from oss.gaussian.canvas.canvas import PersistentCanvas
# from oss.gaussian.canvas.warp import warp_canvas
#
# __all__ = ["PersistentCanvas", "warp_canvas"]

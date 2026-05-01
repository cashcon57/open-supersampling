"""OSS-Gaussian frame extrapolation (Sprint 6).

`FrameExtrapolator` renders the persistent Gaussian canvas at fractional
time positions ``t-1 + alpha`` for any ``alpha ∈ [0, 1]``.

The killer property — and the reason this module is ~150 lines instead of a
training pipeline — is that **frame extrapolation is the same operation as
the Sprint 5 motion warp, parameterised by a smaller alpha**. There is no
separate frame-generation network (cf. DLSS Frame Generation, which is an
additive heavy pass over a learned optical-flow net).

Algorithm (per design spec §3.2 row 6 / master plan Sprint 6):

    1. Take the persistent canvas at time t (Sprint 5 output).
    2. Take motion vectors describing the t-1 → t displacement.
    3. Compute warped_canvas = warp_canvas(canvas, motion, alpha=alpha).
       For alpha = 0 → no shift, output equals canvas.render() at time t.
       For alpha = 1 → full shift, positions match the predicted t+1
       (interpreting the motion field as the t → t+1 vector instead).
    4. Render warped_canvas through the existing Rasterizer at native
       resolution. Done.

The cost above-and-beyond a normal canvas render is one in-place add on
the (N, 2) position tensor — i.e. essentially free relative to the
rasterizer pass that already dominates frame time.

Sprint 5 dependency
-------------------
This module imports `oss.gaussian.canvas.PersistentCanvas` and
`oss.gaussian.canvas.warp_canvas` lazily inside `extrapolate(...)` so the
module can be imported (and most tests can run) before Sprint 5 lands.
If Sprint 5 has not yet shipped, calling `extrapolate(...)` raises
`RuntimeError` with a message pointing to the missing symbols.

For unit testing without Sprint 5, callers may pass a custom
``warp_fn`` to the constructor. The synthetic-motion tests in
``tests/gaussian/test_extrapolation.py`` use this hook so coverage is
not blocked on Sprint 5.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Tuple

import torch

from oss.gaussian.renderer import GaussianBatch, Rasterizer


class _CanvasLike(Protocol):
    """Structural type for what Sprint 5's PersistentCanvas exposes.

    Kept narrow so test doubles can satisfy it without inheriting.
    """

    gaussians: GaussianBatch  # current Gaussians at time t


WarpFn = Callable[[Any, torch.Tensor, float], Any]
"""Signature of a canvas-warp function: (canvas, motion, alpha) -> canvas."""


def _default_warp_fn(canvas: Any, motion: torch.Tensor, alpha: float) -> Any:
    """Lazily-resolved default that delegates to Sprint 5's `warp_canvas`.

    Imported at call time so the module is importable on machines where
    Sprint 5 hasn't been merged yet.
    """
    try:
        # TODO(sprint-5): once canvas.warp.warp_canvas exists this import
        # will resolve. Keep the path matching canvas/__init__.py contract.
        from oss.gaussian.canvas import warp_canvas  # type: ignore[attr-defined]
    except ImportError as e:  # pragma: no cover — exercised only pre-Sprint-5
        raise RuntimeError(
            "FrameExtrapolator: oss.gaussian.canvas.warp_canvas is not "
            "available. Sprint 5 (persistent canvas + warp) must land before "
            "frame extrapolation can run end-to-end. For unit testing, pass "
            "a custom `warp_fn` to FrameExtrapolator."
        ) from e
    return warp_canvas(canvas, motion, alpha=alpha)


class FrameExtrapolator:
    """Render the persistent Gaussian canvas at a fractional time offset.

    Args:
        rasterizer: shared `Rasterizer` instance. If None a default one
            is constructed (auto-selects CUDA / reference backend).
        warp_fn: callable matching `WarpFn`. Defaults to `warp_canvas`
            from Sprint 5's `oss.gaussian.canvas`. Override in tests
            or for ablations.

    Inputs to `extrapolate(...)`:
        canvas: PersistentCanvas at time t (Sprint 5 output).
        motion: (2, H, W) motion field. Convention matches `warp_canvas`
            — see `oss/gaussian/canvas/__init__.py`.
        alpha:  float in [0, 1]. 0 → identity render, 1 → full warp.
        output_hw: (H, W) target resolution for the rendered frame.

    Returns the rendered (F, H, W) tensor — the predicted frame at t+α.
    """

    def __init__(
        self,
        rasterizer: Optional[Rasterizer] = None,
        warp_fn: Optional[WarpFn] = None,
    ) -> None:
        self.rasterizer = rasterizer if rasterizer is not None else Rasterizer()
        self.warp_fn: WarpFn = warp_fn if warp_fn is not None else _default_warp_fn

    def extrapolate(
        self,
        canvas: _CanvasLike,
        motion: torch.Tensor,
        alpha: float,
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Render the canvas at time t-1 + alpha."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must lie in [0, 1]; got {alpha}")
        if motion.ndim != 3 or motion.shape[0] != 2:
            raise ValueError(
                f"motion must have shape (2, H, W); got {tuple(motion.shape)}"
            )
        h, w = output_hw
        if h <= 0 or w <= 0:
            raise ValueError(f"output_hw must be positive; got {output_hw}")

        warped = self.warp_fn(canvas, motion, alpha)
        gaussians = warped.gaussians  # see _CanvasLike protocol
        return self.rasterizer(gaussians, output_hw=output_hw)


__all__ = ["FrameExtrapolator", "WarpFn"]

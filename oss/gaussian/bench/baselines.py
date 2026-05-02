"""Baseline upscalers for OSS-Gaussian comparison benchmarks.

Per the research synthesis (docs/superpowers/research-synthesis-2026-05-01.md
section 5, plan update 1), the graduation criterion against OSSPico alone is
too friendly. Iso-latency comparison vs FSR 2 / DLSS Quality is the
research-grade target.

This module provides baseline upscalers behind a uniform interface so the
Gaussian track can be benchmarked head-to-head:

- ``BicubicUpscaler``        — torch.nn.functional, no engine dependency
- ``LanczosUpscaler``         — pillow / kornia, no engine dependency
- ``FSR2Upscaler``           — wraps the AMD FidelityFX FSR 2 SDK
- ``DLSSQualityUpscaler``    — wraps the NVIDIA NGX SDK DLSS-SR Quality
- ``DLSSFrameGenUpscaler``   — wraps NGX DLSS Frame Generation (extrap baseline)

Only Bicubic + Lanczos are implemented in v0 (no engine dep). FSR2/DLSS are
declared as classes with build-time gates and SDK-presence checks; the
actual SDK wiring lands during Sprint 4 close-out + Sprint 6 measurement
work on the 3080 Ti.

Usage:
    from oss.gaussian.bench.baselines import bicubic_upscale
    hr = bicubic_upscale(lr_frame, scale=2.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class UpscaleResult:
    """Output of a baseline upscaler.

    image: (B, 3, H_hr, W_hr) float tensor in [0, 1] (or HDR if input was HDR).
    elapsed_ms: wall-clock time of the upscaling forward pass, including
        warmup-amortised. Caller should average over a benchmark loop.
    name: identifier for reporting.
    """
    image: torch.Tensor
    elapsed_ms: float
    name: str


class Upscaler(ABC):
    """Uniform interface every baseline (and OSS-Gaussian itself) implements."""

    name: str = "abstract"

    @abstractmethod
    def __call__(
        self,
        lr_frame: torch.Tensor,
        scale: float,
        *,
        depth: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
    ) -> UpscaleResult:
        """Upscale ``lr_frame`` to scale*input resolution.

        Args:
            lr_frame: (B, 3, H, W) low-res RGB.
            scale: 2.0, 3.0, 4.0 typical.
            depth, motion: optional G-buffers; ignored by trivial baselines.
        """


# ---------------------------------------------------------------------------
# Trivial baselines (no engine dependency)
# ---------------------------------------------------------------------------

class BicubicUpscaler(Upscaler):
    name = "bicubic"

    def __call__(self, lr_frame, scale, *, depth=None, motion=None):
        h, w = lr_frame.shape[-2:]
        out = F.interpolate(
            lr_frame, size=(int(h * scale), int(w * scale)),
            mode="bicubic", align_corners=False,
        )
        # We don't record latency here — caller wraps with a torch.cuda.Event
        # or perf_counter loop and updates UpscaleResult.elapsed_ms.
        return UpscaleResult(image=out, elapsed_ms=0.0, name=self.name)


class LanczosUpscaler(Upscaler):
    name = "lanczos"

    def __call__(self, lr_frame, scale, *, depth=None, motion=None):
        # PyTorch doesn't ship Lanczos. Use kornia if available; fall back to
        # a manual area+bicubic combo.
        try:
            import kornia  # type: ignore[import-untyped]
            h, w = lr_frame.shape[-2:]
            out = kornia.geometry.transform.resize(
                lr_frame, (int(h * scale), int(w * scale)),
                interpolation="lanczos", align_corners=False,
            )
        except Exception:
            # Fallback: bicubic. Documented; produces the same output as
            # BicubicUpscaler — caller should be aware kornia isn't installed.
            return BicubicUpscaler()(lr_frame, scale, depth=depth, motion=motion)
        return UpscaleResult(image=out, elapsed_ms=0.0, name=self.name)


# Convenience top-level functions
def bicubic_upscale(lr_frame: torch.Tensor, scale: float) -> torch.Tensor:
    """Shortcut returning the upscaled image only."""
    return BicubicUpscaler()(lr_frame, scale).image


def lanczos_upscale(lr_frame: torch.Tensor, scale: float) -> torch.Tensor:
    return LanczosUpscaler()(lr_frame, scale).image


# ---------------------------------------------------------------------------
# Vendor SDK baselines (deferred wiring)
# ---------------------------------------------------------------------------

class FSR2Upscaler(Upscaler):
    """AMD FidelityFX Super Resolution 2.

    Requires the FSR 2 SDK at build time + Vulkan/D3D12 context at runtime.
    On the OSS-Gaussian dev machine (M3 Max), this raises NotImplementedError.
    On the 3080 Ti, Sprint 4 close-out wires it via the Vulkan/D3D12 host
    harness in ``oss/gaussian/interception/`` (Sprint 2 builds the harness).
    """

    name = "fsr2_quality"

    def __call__(self, lr_frame, scale, *, depth=None, motion=None):
        raise NotImplementedError(
            "FSR2Upscaler not yet wired. Requires the AMD FidelityFX FSR 2 "
            "SDK and a Vulkan/D3D12 host context. Sprint 4 close-out task "
            "T4.13 (added by research-synthesis plan update) wires this."
        )


class DLSSQualityUpscaler(Upscaler):
    """NVIDIA DLSS Super Resolution at the Quality preset.

    Requires the NGX SDK + an NVIDIA RTX GPU at runtime. On the M3 Max this
    raises NotImplementedError. The 3080 Ti has the GPU; the SDK wiring
    happens during Sprint 2 (which already builds the NGX shim DLL).
    """

    name = "dlss_sr_quality"

    def __call__(self, lr_frame, scale, *, depth=None, motion=None):
        raise NotImplementedError(
            "DLSSQualityUpscaler not yet wired. Requires the NVIDIA NGX SDK "
            "and an RTX GPU. Sprint 2 (D3D12 hook) provides the NGX harness; "
            "this class will route a Quality-preset eval call through it."
        )


class DLSSFrameGenUpscaler(Upscaler):
    """NVIDIA DLSS Frame Generation. Used as the comparison baseline for
    Sprint 6 (frame extrapolation).

    Note: DLSS-FG interpolates between t and t+1, adding one frame of
    latency. OSS-Gaussian extrapolates from t to t+α, no added latency.
    The comparison is asymmetric — quality vs latency tradeoff that we
    report explicitly, not collapse to a single number.
    """

    name = "dlss_fg"

    def __call__(self, lr_frame, scale, *, depth=None, motion=None):
        raise NotImplementedError(
            "DLSSFrameGenUpscaler not yet wired. Requires NGX DLSS-FG. "
            "Sprint 6 integrates this for the frame-gen comparison report."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[Upscaler]] = {
    "bicubic": BicubicUpscaler,
    "lanczos": LanczosUpscaler,
    "fsr2_quality": FSR2Upscaler,
    "dlss_sr_quality": DLSSQualityUpscaler,
    "dlss_fg": DLSSFrameGenUpscaler,
}


def make(name: str) -> Upscaler:
    """Factory: build a baseline by name. Raises KeyError for unknown."""
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = [
    "Upscaler",
    "UpscaleResult",
    "BicubicUpscaler",
    "LanczosUpscaler",
    "FSR2Upscaler",
    "DLSSQualityUpscaler",
    "DLSSFrameGenUpscaler",
    "REGISTRY",
    "make",
    "bicubic_upscale",
    "lanczos_upscale",
]

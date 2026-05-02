"""Engine-aliased LR synthesis pipeline for OSS-Gaussian Sprint 4.

WHY THIS MODULE EXISTS
----------------------
Training SR networks against bicubic-clean LR images creates a
"bicubic-LR-trap": bicubic upsampling is the near-inverse of bicubic
downsampling, so any SR network that hallucinates HF detail will *lose* PSNR
on clean-bicubic benchmarks. Real-ESRGAN §3.1 documents this problem
explicitly.

The 2026-05-01 validation decision memo (Decision 3) mandates that Sprint 4
training data must be **engine-aliased** — the LR side of each training pair
must look like what a real game engine actually emits, not what a bicubic
filter produces:

    1. Per-frame Halton-2 subpixel jitter at HR before downsample
       → matches real TAA jitter offsets (Halton(2,3) sequence)
    2. Area-filter downsample (box average, integer-divisor)
       → avoids bicubic ringing
    3. TAA blur simulation (3×3 Gaussian, σ≈0.5)
       → models the temporal blur introduced by TAA's exponential moving
         average across frames (full EMA would require prior-frame warping
         which is impractical in a random-access dataset loader)
    4. Optional JPEG artifacts at quality 85
       → models content-delivery scenarios (streaming, compressed capture)

Public entry point:  ``EngineAliasedLRSynth.synthesize(hr, frame_idx)``
Individual helpers are exposed for testing and composition.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ==============================================================================
# 1.  Halton low-discrepancy sequence
# ==============================================================================


def _halton_single(idx: int, base: int) -> float:
    """Return the idx-th element of the Van der Corput sequence in ``base``.

    The Van der Corput sequence is the 1-D building block of Halton sequences.
    It provides low-discrepancy quasi-random values in [0, 1) that cover the
    unit interval more uniformly than pseudo-random numbers, making it a good
    model for TAA jitter patterns (which real engines implement the same way).
    """
    result = 0.0
    denominator = 1.0
    n = idx
    while n > 0:
        denominator *= base
        n, remainder = divmod(n, base)
        result += remainder / denominator
    return result


def halton_jitter(idx: int, base_x: int = 2, base_y: int = 3) -> tuple[float, float]:
    """Return the (jx, jy) sub-pixel jitter offset for ``frame_idx`` in HR pixels.

    Uses the Halton(base_x, base_y) quasi-random sequence — the same construction
    used by Unreal Engine and DLSS for TAA jitter patterns.  The offset is
    shifted from [0, 1) to [-0.5, 0.5) so that jitter is symmetric around the
    pixel centre.

    The sequence is evaluated at ``idx + 1`` rather than ``idx`` to match the
    TAA convention used by Unreal Engine and DLSS: Halton(0, base) == 0.0, which
    maps to the maximum-offset corner (-0.5, -0.5) after the [-0.5, 0.5) shift.
    Starting at index 1 avoids this corner-bias on the first frame of every epoch
    and keeps the sequence consistent with real engine implementations.

    Args:
        idx:    frame index (0-based, monotonically increasing per sequence).
        base_x: Halton base for the x-axis jitter. Default: 2.
        base_y: Halton base for the y-axis jitter. Default: 3.

    Returns:
        (jx, jy) in HR-pixel units, each in [-0.5, 0.5].
    """
    jx = _halton_single(idx + 1, base_x) - 0.5
    jy = _halton_single(idx + 1, base_y) - 0.5
    return float(jx), float(jy)


# ==============================================================================
# 2.  Sub-pixel shift via bilinear grid_sample
# ==============================================================================


def apply_jitter(hr: Tensor, jitter: tuple[float, float]) -> Tensor:
    """Shift a (C, H, W) HR frame by a sub-pixel offset using bilinear sampling.

    The shift is expressed in HR pixel units.  Positive jx shifts the content
    to the right (the view shifts left); positive jy shifts down.

    ``F.grid_sample`` operates on a normalised grid in [-1, 1], so we convert
    the pixel-unit offset to normalised coordinates before building the grid.

    Args:
        hr:     (C, H, W) float32 tensor in [0, 1].
        jitter: (jx, jy) offset in HR pixel units, each in [-0.5, 0.5].

    Returns:
        (C, H, W) float32 tensor — the bilinearly-shifted image.
    """
    if hr.dim() != 3:
        raise ValueError(f"apply_jitter expects (C, H, W); got {tuple(hr.shape)}")

    jx, jy = jitter
    C, H, W = hr.shape

    # grid_sample normalised coords: [-1, 1] spans the full image width/height.
    # A positive normalised-x shift moves the sampling grid right (content left).
    # We want a pixel-unit jx to shift the *content* right, so the grid moves
    # left: norm_x_shift = -jx * 2 / W (per column).
    norm_dx = -jx * 2.0 / W
    norm_dy = -jy * 2.0 / H

    # Base identity grid: shape (1, H, W, 2), values in [-1, 1].
    base = F.affine_grid(
        torch.eye(2, 3, dtype=hr.dtype).unsqueeze(0),  # identity transform
        size=(1, C, H, W),
        align_corners=False,
    )  # (1, H, W, 2): [..., 0] = x coord, [..., 1] = y coord

    # Apply uniform shift.
    shifted = base.clone()
    shifted[..., 0] += norm_dx
    shifted[..., 1] += norm_dy

    out = F.grid_sample(
        hr.unsqueeze(0),   # (1, C, H, W)
        shifted,           # (1, H, W, 2)
        mode="bilinear",
        padding_mode="border",  # replicate border rather than zero-pad
        align_corners=False,
    )
    return out.squeeze(0)  # (C, H, W)


# ==============================================================================
# 3.  Area downsample (box average, integer-divisor)
# ==============================================================================


def area_downsample(hr: Tensor, scale: float) -> Tensor:
    """Box-average downsample (C, H, W) → LR using integer-pixel divisor.

    Mathematically identical to ``GaussianDataset._box_downsample`` — kept as a
    standalone function so it can be imported from this module and used without
    the base class.  The implementation is deliberately kept in sync with
    ``_box_downsample`` so that regression test 3 passes (byte-for-byte match).

    Args:
        hr:    (C, H, W) float32 HR tensor.
        scale: integer-ish downsample factor (e.g. 2.0, 4.0).

    Returns:
        (C, H//s, W//s) float32 LR tensor.
    """
    if hr.dim() != 3:
        raise ValueError(f"area_downsample expects (C, H, W); got {tuple(hr.shape)}")
    s = int(round(scale))
    if s < 1:
        raise ValueError(f"scale must be >=1; got {scale}")
    if s == 1:
        return hr.clone()
    C, H, W = hr.shape
    H2 = (H // s) * s
    W2 = (W // s) * s
    x = hr[:, :H2, :W2]
    x = x.view(C, H2 // s, s, W2 // s, s).mean(dim=(2, 4))
    return x


# ==============================================================================
# 4.  TAA blur approximation (small Gaussian kernel)
# ==============================================================================


def _gaussian_kernel_2d(sigma: float, kernel_size: int = 3) -> Tensor:
    """Build a 2-D Gaussian kernel of the given sigma and size.

    The kernel is separable and normalised to sum to 1.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    kernel = g.outer(g)
    return kernel


def taa_blur_approx(lr: Tensor, sigma: float = 0.5, kernel_size: int = 3) -> Tensor:
    """Simulate TAA temporal blur via a small Gaussian blur on the LR image.

    Real TAA accumulates frames using an exponential moving average (EMA):
        I_t = α·I_t + (1−α)·warp(I_{t−1})
    This EMA blurs sharp edges across time.  In a random-access dataset where
    we do not have access to the prior frame and its optical flow, we approximate
    the effect with a single-frame Gaussian blur.

    Args:
        lr:          (C, H, W) float32 LR tensor.
        sigma:       Gaussian standard deviation in LR pixels. Default 0.5
                     (mild — matches a 10-frame EMA at α=0.1). Bump to 1.0–1.5
                     for more aggressive engine-aliased degradation.
        kernel_size: Odd kernel side length. Default 3. Must satisfy
                     ``3*sigma ≤ kernel_size // 2`` to avoid clipping the tail.

    Returns:
        (C, H, W) float32 blurred LR tensor.
    """
    if lr.dim() != 3:
        raise ValueError(f"taa_blur_approx expects (C, H, W); got {tuple(lr.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd; got {kernel_size}")
    C, H, W = lr.shape

    kernel = _gaussian_kernel_2d(sigma=sigma, kernel_size=kernel_size)  # (k, k)
    kernel = kernel.to(lr.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    out = F.conv2d(
        lr.unsqueeze(0),
        kernel,
        padding=pad,
        groups=C,
    ).squeeze(0)

    return out.clamp(0.0, 1.0)


# ==============================================================================
# 5.  Optional JPEG artifact round-trip
# ==============================================================================


def jpeg_artifact(lr: Tensor, quality: int = 85) -> Tensor:
    """Apply a JPEG encode → decode round-trip to introduce compression artefacts.

    This models content-delivery scenarios where game captures are JPEG-compressed
    (e.g., streaming, screenshot pipelines).  Default quality 85 is typical for
    high-quality JPEG delivery.

    Implementation uses PIL for the round-trip, which is guaranteed available
    via torchvision's dependency chain.  We avoid torchvision.io.encode_jpeg
    because it requires uint8 tensors and a JPEG codec compiled into libtorch,
    which is not guaranteed in all environments.

    Args:
        lr:      (C, H, W) float32 tensor in [0, 1].
        quality: JPEG quality factor in [1, 95]. Default: 85.

    Returns:
        (C, H, W) float32 tensor after JPEG round-trip, values in [0, 1].
    """
    if lr.dim() != 3:
        raise ValueError(f"jpeg_artifact expects (C, H, W); got {tuple(lr.shape)}")

    from PIL import Image

    C, H, W = lr.shape
    # PIL expects (H, W, C) uint8.
    uint8 = (lr.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(uint8, mode="RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_decoded = Image.open(buf).convert("RGB")

    import numpy as np  # already in deps
    arr = torch.from_numpy(np.array(pil_decoded, dtype="float32") / 255.0)
    # arr is (H, W, C); convert to (C, H, W).
    result = arr.permute(2, 0, 1).to(lr.dtype)
    return result.clamp(0.0, 1.0)


# ==============================================================================
# 6.  EngineAliasedLRSynth orchestrator
# ==============================================================================


@dataclass
class EngineAliasedLRSynth:
    """Orchestrates the engine-aliased LR synthesis pipeline.

    Each component is independently toggleable so that ablation experiments can
    isolate the contribution of individual degradation modes.

    Pipeline (applied in order when enabled):
        1. Halton-2 subpixel jitter on HR
        2. Area-filter downsample (always applied)
        3. TAA blur approximation on LR
        4. JPEG artefacts on LR (optional)

    Args:
        scale:          HR/LR scale factor (must be an integer-ish value ≥ 1).
        enable_jitter:  Apply per-frame Halton subpixel jitter. Default: True.
        enable_taa_blur: Apply 3×3 Gaussian TAA blur approximation. Default: True.
        enable_jpeg:    Apply JPEG compression artefacts. Default: False.
        jpeg_quality:   JPEG quality factor when enable_jpeg is True. Default: 85.

    Usage::

        synth = EngineAliasedLRSynth(scale=2.0)
        lr = synth.synthesize(hr_tensor, frame_idx=frame_counter)
    """

    scale: float = 2.0
    enable_jitter: bool = True
    enable_taa_blur: bool = True
    enable_jpeg: bool = False
    jpeg_quality: int = 85
    blur_sigma: float = 0.5  # TAA blur kernel sigma; raise to 1.0–1.5 for more aggressive degradation

    def __post_init__(self) -> None:
        if self.scale < 1.0:
            raise ValueError(f"scale must be >=1.0; got {self.scale}")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(f"jpeg_quality must be in [1, 95]; got {self.jpeg_quality}")
        if self.blur_sigma <= 0:
            raise ValueError(f"blur_sigma must be > 0; got {self.blur_sigma}")

    def synthesize(self, hr: Tensor, frame_idx: int) -> Tensor:
        """Produce an engine-aliased LR frame from a high-resolution input.

        Args:
            hr:        (C, H, W) float32 HR tensor in [0, 1].
            frame_idx: Monotonically increasing frame index used to index the
                       Halton sequence for jitter.  Treat as a per-dataset-item
                       index in random-access loaders.

        Returns:
            (C, H//scale, W//scale) float32 LR tensor in [0, 1].
        """
        if hr.dim() != 3:
            raise ValueError(f"synthesize expects (C, H, W); got {tuple(hr.shape)}")

        source = hr

        # Step 1 — Halton subpixel jitter on HR (before downsample).
        if self.enable_jitter:
            jitter = halton_jitter(frame_idx)
            source = apply_jitter(source, jitter)

        # Step 2 — Area-filter downsample (always applied).
        lr = area_downsample(source, self.scale)

        # Step 3 — TAA blur approximation on LR.
        if self.enable_taa_blur:
            # Pick kernel size so 3*sigma fits inside the kernel half-width.
            ksize = max(3, int(2 * round(3 * self.blur_sigma) + 1))
            lr = taa_blur_approx(lr, sigma=self.blur_sigma, kernel_size=ksize)

        # Step 4 — Optional JPEG artefacts.
        if self.enable_jpeg:
            lr = jpeg_artifact(lr, quality=self.jpeg_quality)

        return lr.contiguous().float()


__all__ = [
    "halton_jitter",
    "apply_jitter",
    "area_downsample",
    "taa_blur_approx",
    "jpeg_artifact",
    "EngineAliasedLRSynth",
]

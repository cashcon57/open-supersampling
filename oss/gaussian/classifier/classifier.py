"""OSS-Gaussian Sprint 3 — Tile Classifier.

Per-frame heuristic that classifies each 16x16 tile of a low-resolution
frame as 'complex' (Sprint 4 network must predict Gaussian params) or
'simple' (bilinear passthrough; bypass the network).

The classifier is a pure PyTorch module — runs unchanged on CPU and CUDA.
A custom CUDA kernel is deferred (master plan: out-of-scope for v1; follow-up
sprint task if perf budgets are missed).

Inputs (NCHW, B-major, float32):
- frame:   (B, 3, H, W)   LR RGB
- depth:   (B, 1, H, W)   LR linear depth (any units; ratios used)
- motion:  (B, 2, H, W)   LR per-pixel motion vectors (px or NDC)
- normals: (B, 3, H, W)   Optional. Unit-length world/view normals

Output:
- mask:    (B, H/T, W/T)  bool. True = complex.

H, W must be multiples of `tile_size`.

Spec: docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md (3.1, 3.2 row 3)
Plan: docs/superpowers/plans/2026-05-01-gaussian-sprint-3-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Match the renderer's hard-coded tile size. See
# `oss/gaussian/renderer/rasterizer.py:TILE_SIZE`.
DEFAULT_TILE_SIZE: int = 16

# BT.709 luma weights — used to collapse RGB to a single channel for gradient
# computation. Matches what most game engines use for tonemapping inputs.
_LUMA_WEIGHTS: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class FeatureWeights:
    """Per-feature weights for combining the per-tile complexity score.

    Defaults picked from T3.6 ablation (placeholder values until Sprint 2
    Cyberpunk frames land; see plan T3.6 for re-calibration).
    """

    rgb_grad: float = 1.0
    depth_disc: float = 1.0
    motion: float = 0.5
    normal_var: float = 0.25


class TileClassifier:
    """Classify each 16x16 tile of an LR frame as complex or simple.

    Args:
        tile_size: edge length in pixels (default 16 — matches renderer).
        target_complex_fraction: fraction of tiles to mark complex per frame.
            Adaptive threshold finds the matching score cutoff via ``kthvalue``.
        weights: feature weights for the composite complexity score.
        eps: numerical floor used in log(depth) and per-feature normalization.
    """

    def __init__(
        self,
        tile_size: int = DEFAULT_TILE_SIZE,
        target_complex_fraction: float = 0.30,
        weights: Optional[FeatureWeights] = None,
        eps: float = 1e-6,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive; got {tile_size}")
        if not 0.0 <= target_complex_fraction <= 1.0:
            raise ValueError(
                f"target_complex_fraction must be in [0, 1]; got {target_complex_fraction}"
            )
        self.tile_size = int(tile_size)
        self.target_complex_fraction = float(target_complex_fraction)
        self.weights = weights if weights is not None else FeatureWeights()
        self.eps = float(eps)

    # ------------------------------------------------------------------ public

    def __call__(
        self,
        frame: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a (B, H/T, W/T) bool mask. True = complex tile."""
        self._validate(frame, depth, motion, normals)
        score = self.score(frame, depth, motion, normals)
        return self._threshold(score)

    def score(
        self,
        frame: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the per-tile complexity score, (B, H/T, W/T) float32.

        Exposed publicly for the visualization helper, ablation tooling, and
        debugging — the threshold step is just `score > kthvalue`.
        """
        self._validate(frame, depth, motion, normals)
        T = self.tile_size

        rgb_grad = self._tile_rgb_gradient(frame, T)        # (B, h, w)
        depth_disc = self._tile_depth_discontinuity(depth, T)
        motion_mag = self._tile_motion_magnitude(motion, T)

        feats = [
            (rgb_grad, self.weights.rgb_grad),
            (depth_disc, self.weights.depth_disc),
            (motion_mag, self.weights.motion),
        ]
        if normals is not None:
            feats.append((self._tile_normal_variance(normals, T), self.weights.normal_var))

        # Per-frame, per-feature normalization avoids one feature with larger
        # raw units dominating the composite score.
        score = torch.zeros_like(rgb_grad)
        for feat, w in feats:
            score = score + w * self._normalize_per_frame(feat)
        return score

    # ---------------------------------------------------------------- features

    def _tile_rgb_gradient(self, frame: torch.Tensor, T: int) -> torch.Tensor:
        """Mean Sobel gradient magnitude on luma, reduced per tile."""
        r, g, b = _LUMA_WEIGHTS
        luma = r * frame[:, 0:1] + g * frame[:, 1:2] + b * frame[:, 2:3]  # (B,1,H,W)
        gx, gy = _sobel(luma)
        grad_mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        return F.avg_pool2d(grad_mag, kernel_size=T, stride=T).squeeze(1)

    def _tile_depth_discontinuity(self, depth: torch.Tensor, T: int) -> torch.Tensor:
        """Max gradient magnitude on log(depth), reduced per tile.

        log() makes the metric scale-invariant (same response at near and far
        depth steps). max() is used because a single hard edge anywhere inside
        a tile must trigger the tile complex.
        """
        log_d = torch.log(depth.clamp_min(self.eps))
        gx, gy = _sobel(log_d)
        disc = torch.sqrt(gx * gx + gy * gy + self.eps)
        return F.max_pool2d(disc, kernel_size=T, stride=T).squeeze(1)

    def _tile_motion_magnitude(self, motion: torch.Tensor, T: int) -> torch.Tensor:
        """Mean motion vector magnitude per tile."""
        mag = torch.sqrt(motion[:, 0:1] ** 2 + motion[:, 1:2] ** 2 + self.eps)
        return F.avg_pool2d(mag, kernel_size=T, stride=T).squeeze(1)

    def _tile_normal_variance(self, normals: torch.Tensor, T: int) -> torch.Tensor:
        """Per-tile angular variance of unit normals.

        Computed as 1 - ||mean(n)||_2 over the tile. For aligned unit vectors,
        the tile-mean has length 1; for randomly-oriented vectors, length -> 0.
        """
        mean_n = F.avg_pool2d(normals, kernel_size=T, stride=T)  # (B,3,h,w)
        mean_len = torch.sqrt((mean_n * mean_n).sum(dim=1) + self.eps)  # (B,h,w)
        return (1.0 - mean_len).clamp_min(0.0)

    # ----------------------------------------------------------------- helpers

    def _normalize_per_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each batch element by its own max so features are commensurate."""
        b = x.shape[0]
        flat = x.reshape(b, -1)
        denom = flat.amax(dim=1).clamp_min(self.eps).reshape(b, 1, 1)
        return x / denom

    def _threshold(self, score: torch.Tensor) -> torch.Tensor:
        """Pick a per-frame threshold so target_complex_fraction of tiles fire."""
        b, h, w = score.shape
        n = h * w
        f = self.target_complex_fraction

        if f <= 0.0:
            return torch.zeros_like(score, dtype=torch.bool)
        if f >= 1.0:
            return torch.ones_like(score, dtype=torch.bool)

        # k-th smallest, with k chosen so that (n - k + 1) tiles end up >= thresh.
        # That is: target_count = round(f * n);  k = n - target_count + 1.
        # Using `>=` (rather than `>`) plus this k keeps the actual fraction
        # closest to target when scores have ties at the threshold.
        target_count = max(1, min(n, int(round(f * n))))
        k = n - target_count + 1  # in [1, n]
        flat = score.reshape(b, n)
        # kthvalue is supported on CPU and CUDA in torch >= 1.8.
        thresh = flat.kthvalue(k, dim=1).values.reshape(b, 1, 1)
        return score >= thresh

    def _validate(
        self,
        frame: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        normals: Optional[torch.Tensor],
    ) -> None:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(f"frame must be (B,3,H,W); got {tuple(frame.shape)}")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"depth must be (B,1,H,W); got {tuple(depth.shape)}")
        if motion.ndim != 4 or motion.shape[1] != 2:
            raise ValueError(f"motion must be (B,2,H,W); got {tuple(motion.shape)}")
        b, _, h, w = frame.shape
        if depth.shape[0] != b or motion.shape[0] != b:
            raise ValueError(
                f"batch sizes disagree: frame={frame.shape[0]}, depth={depth.shape[0]}, motion={motion.shape[0]}"
            )
        if depth.shape[-2:] != (h, w) or motion.shape[-2:] != (h, w):
            raise ValueError(
                f"spatial dims disagree: frame={(h, w)}, depth={tuple(depth.shape[-2:])}, motion={tuple(motion.shape[-2:])}"
            )
        T = self.tile_size
        if h % T or w % T:
            raise ValueError(
                f"H={h} and W={w} must be multiples of tile_size={T}"
            )
        if normals is not None:
            if normals.ndim != 4 or normals.shape[1] != 3:
                raise ValueError(f"normals must be (B,3,H,W); got {tuple(normals.shape)}")
            if normals.shape[0] != b or normals.shape[-2:] != (h, w):
                raise ValueError(
                    f"normals shape mismatch with frame: {tuple(normals.shape)} vs {tuple(frame.shape)}"
                )


# ---------------------------------------------------------------- module-free


def _sobel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradient via depthwise conv2d. x: (B,1,H,W) → (gx, gy) same shape.

    Reflection padding is used to keep tile boundary tiles meaningful (avoids
    zero-pad creating artificial edges at the frame border).
    """
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(pad, kx)
    gy = F.conv2d(pad, ky)
    return gx, gy


def overlay_mask(
    frame: torch.Tensor,
    mask: torch.Tensor,
    tile_size: int = DEFAULT_TILE_SIZE,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compose a debug visualization: complex tiles tinted ``color`` over frame.

    Args:
        frame: (B, 3, H, W) float in any range. Output keeps the input's
            numeric range — caller is responsible for tone-mapping if needed.
        mask:  (B, H/T, W/T) bool. True tiles get tinted.
        tile_size: must match the size used to produce ``mask``.
        color: RGB tint, each channel in [0, 1].
        alpha: blend factor in [0, 1] (0 = no tint, 1 = full color).

    Returns:
        (B, 3, H, W) float, same dtype/device as ``frame``.
    """
    if frame.ndim != 4 or frame.shape[1] != 3:
        raise ValueError(f"frame must be (B,3,H,W); got {tuple(frame.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be (B,h,w); got {tuple(mask.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    b, _, h, w = frame.shape
    expected_h = mask.shape[1] * tile_size
    expected_w = mask.shape[2] * tile_size
    if expected_h != h or expected_w != w:
        raise ValueError(
            f"mask {tuple(mask.shape)} * tile_size={tile_size} = ({expected_h},{expected_w}) "
            f"does not match frame ({h},{w})"
        )

    # Upsample mask to pixel resolution by nearest-neighbor.
    mask_px = (
        mask.to(frame.dtype)
        .unsqueeze(1)  # (B,1,h,w)
        .repeat_interleave(tile_size, dim=2)
        .repeat_interleave(tile_size, dim=3)
    )  # (B,1,H,W)

    color_tensor = torch.tensor(color, dtype=frame.dtype, device=frame.device).view(1, 3, 1, 1)
    tint = color_tensor.expand(b, 3, h, w)
    blended = frame * (1.0 - alpha * mask_px) + tint * (alpha * mask_px)
    return blended


__all__ = [
    "TileClassifier",
    "FeatureWeights",
    "DEFAULT_TILE_SIZE",
    "overlay_mask",
]

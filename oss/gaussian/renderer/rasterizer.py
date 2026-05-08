"""OSS-Gaussian renderer: thin Python wrapper over the vendored Image-GS / gsplat
2D Gaussian rasterizer.

The wrapper provides:
- Stable typed API independent of upstream changes in Image-GS.
- A reference PyTorch fallback for environments without the CUDA extension
  (development on Mac M-series before the M3 Max Metal port lands).
- Inputs validated at the boundary so the CUDA path receives well-formed tensors.

Sprint 1 deliverable. Wired into:
- T1.4 forward render test (tests/gaussian/test_renderer_forward.py)
- T1.5 differentiable backward test (tests/gaussian/test_renderer_backward.py)
- T1.6 performance benchmark (oss/gaussian/renderer/bench.py)
- T1.7 integration smoke test (tests/gaussian/test_renderer_integration.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

# gsplat is provided by the vendored Image-GS submodule. It builds CUDA kernels
# at install time. We import lazily so that machines without CUDA can still
# import this module and use the reference fallback.
_GSPLAT_AVAILABLE: bool = False
_GSPLAT_IMPORT_ERROR: str | None = None
try:
    from gsplat import (  # type: ignore[import-not-found]
        project_gaussians_2d_scale_rot,
        rasterize_gaussians_sum,
    )

    _GSPLAT_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — import failure is captured for diagnostics
    _GSPLAT_IMPORT_ERROR = repr(_e)


# Hardcoded in upstream Image-GS CUDA kernel — must not change without modifying
# the kernel source. See `vendor/image_gs/model.py:198` (`self.block_h, self.block_w = 16, 16`).
TILE_SIZE: int = 16
CUDA_MAX_CHANNELS: int = 12


@dataclass(frozen=True)
class GaussianBatch:
    """A batch of 2D Gaussians.

    Shapes use N for the number of Gaussians. The feature channel dimension F
    is variable (1 for grayscale, 3 for RGB, more for stacked feature maps).
    All tensors live on the same device and dtype.

    Attributes:
        xy:    (N, 2)   pixel-space positions in [0, img_w] × [0, img_h]
        scale: (N, 2)   per-axis scale factors, positive
        rot:   (N,)     rotation angle in radians [0, π]
        feat:  (N, F)   per-Gaussian feature/color values
    """

    xy: torch.Tensor
    scale: torch.Tensor
    rot: torch.Tensor
    feat: torch.Tensor

    def __post_init__(self) -> None:
        n = self.xy.shape[0]
        if self.xy.shape != (n, 2):
            raise ValueError(f"xy must be (N, 2), got {tuple(self.xy.shape)}")
        if self.scale.shape != (n, 2):
            raise ValueError(f"scale must be (N, 2), got {tuple(self.scale.shape)}")
        if self.rot.shape != (n,):
            raise ValueError(f"rot must be (N,), got {tuple(self.rot.shape)}")
        if self.feat.ndim != 2 or self.feat.shape[0] != n:
            raise ValueError(f"feat must be (N, F), got {tuple(self.feat.shape)}")
        for name, t in (("xy", self.xy), ("scale", self.scale), ("rot", self.rot), ("feat", self.feat)):
            if t.device != self.xy.device:
                raise ValueError(f"all tensors must share a device; {name} on {t.device}, xy on {self.xy.device}")

    @property
    def num_gaussians(self) -> int:
        return self.xy.shape[0]

    @property
    def feat_dim(self) -> int:
        return self.feat.shape[1]

    @property
    def device(self) -> torch.device:
        return self.xy.device


class Rasterizer:
    """Tile-based top-K 2D Gaussian rasterizer.

    Two implementations:
    - CUDA (preferred): calls the vendored gsplat extension.
    - PyTorch reference (fallback): naive O(N×H×W) implementation in pure PyTorch
      for correctness validation on machines without CUDA. Slow — not for production.

    Selection is automatic based on tensor device and gsplat availability,
    overridable with `force_backend`.

    Args:
        tile_size: tile edge length in pixels. Must equal `TILE_SIZE` (16) for the
            CUDA backend. The reference backend tolerates any value.
        topk_norm: whether to apply top-K weight normalization in the tile
            accumulator. Matches Image-GS default (True).
        force_backend: "cuda" / "reference" / None (auto).
    """

    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        topk_norm: bool = True,
        force_backend: str | None = None,
    ) -> None:
        if force_backend not in (None, "cuda", "reference"):
            raise ValueError(f"force_backend must be 'cuda', 'reference', or None; got {force_backend!r}")
        if tile_size != TILE_SIZE and force_backend == "cuda":
            raise ValueError(
                f"CUDA backend requires tile_size={TILE_SIZE} (hardcoded in kernel); got {tile_size}"
            )
        self.tile_size = tile_size
        self.topk_norm = topk_norm
        self.force_backend = force_backend

    def __call__(
        self,
        gaussians: GaussianBatch,
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Render the Gaussians at the requested output resolution.

        Args:
            gaussians: GaussianBatch on cuda or cpu.
            output_hw: (H, W) target resolution in pixels.

        Returns:
            (F, H, W) image tensor on the same device as `gaussians`.
            F is `gaussians.feat_dim`.
        """
        h, w = output_hw
        if h <= 0 or w <= 0:
            raise ValueError(f"output_hw must be positive; got {output_hw}")
        backend = self._select_backend(gaussians)
        if backend == "cuda":
            return self._render_cuda(gaussians, h, w)
        return self._render_reference(gaussians, h, w)

    def _select_backend(self, gaussians: GaussianBatch) -> str:
        if self.force_backend is not None:
            return self.force_backend
        if gaussians.device.type == "cuda" and _GSPLAT_AVAILABLE:
            return "cuda"
        return "reference"

    def _render_cuda(self, gaussians: GaussianBatch, h: int, w: int) -> torch.Tensor:
        if not _GSPLAT_AVAILABLE:
            raise RuntimeError(
                "CUDA backend requested but gsplat is not importable. "
                f"Original import error: {_GSPLAT_IMPORT_ERROR}. "
                "Build the extension: cd oss/gaussian/renderer/vendor/image_gs && pip install -e ."
            )
        if gaussians.feat_dim > CUDA_MAX_CHANNELS:
            chunks = []
            for feat in gaussians.feat.split(CUDA_MAX_CHANNELS, dim=-1):
                chunk = GaussianBatch(
                    xy=gaussians.xy,
                    scale=gaussians.scale,
                    rot=gaussians.rot,
                    feat=feat.contiguous(),
                )
                chunks.append(self._render_cuda(chunk, h, w))
            return torch.cat(chunks, dim=0).contiguous()

        # gsplat tile_bounds is (num_tiles_W, num_tiles_H, 1) — width first, height
        # second. Reference: oss/gaussian/renderer/vendor/image_gs/model.py:130.
        tile_bounds = ((w + self.tile_size - 1) // self.tile_size,
                       (h + self.tile_size - 1) // self.tile_size,
                       1)
        # gsplat 1.4.0 uses normalized [0, 1] centers but pixel-space scales.
        # Our public API takes both centers and scales in pixel space, so only
        # centers are normalized here.
        # Verified by direct test: xy=(0.25, 0.25) at 64x64 → projected
        # xy_proj=(16, 16), 16 tile hits. xy=(16, 16) → projected (1024, 1024)
        # outside frame → 0 hits → degenerate-tile crash inside the kernel.
        norm = torch.tensor([w, h], dtype=gaussians.xy.dtype, device=gaussians.xy.device)
        xy_norm = gaussians.xy / norm
        scale_px = gaussians.scale.clamp_min(1.0e-3)
        # gsplat expects rot as (N, 1) per Image-GS model.py line 167. Our
        # public API uses (N,) (more natural in PyTorch), so unsqueeze here.
        # Without this the backward path returns gradient shape (N, 1) for
        # an input (N,) tensor → autograd shape-mismatch RuntimeError.
        rot_unsq = gaussians.rot.unsqueeze(-1)
        xy_proj, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            xy_norm, scale_px, rot_unsq, h, w, tile_bounds,
        )
        out_flat = rasterize_gaussians_sum(
            xy_proj, radii, conics, num_tiles_hit,
            gaussians.feat,
            h, w,
            self.tile_size, self.tile_size,
            self.topk_norm,
        )
        # gsplat returns (H*W, F); reshape to (F, H, W).
        return out_flat.view(h, w, gaussians.feat_dim).permute(2, 0, 1).contiguous()

    def _render_reference(
        self,
        gaussians: GaussianBatch,
        h: int,
        w: int,
        conic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Naive PyTorch reference rasterizer.

        For each pixel, accumulate weighted contributions from every Gaussian.
        O(N * H * W) — only suitable for tests on small inputs.
        """
        n = gaussians.num_gaussians
        if n == 0:
            return torch.zeros((gaussians.feat_dim, h, w), device=gaussians.device, dtype=gaussians.feat.dtype)
        if conic is not None and conic.shape != (n, 3):
            raise ValueError(f"conic must be (N, 3), got {tuple(conic.shape)}")

        device = gaussians.device
        ys = torch.arange(h, device=device, dtype=gaussians.xy.dtype)
        xs = torch.arange(w, device=device, dtype=gaussians.xy.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        # (H, W, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        if conic is None:
            # Per-Gaussian covariance from scale + rot.
            cos_t = torch.cos(gaussians.rot)  # (N,)
            sin_t = torch.sin(gaussians.rot)
            # R = [[cos, -sin], [sin, cos]]; S = diag(scale)
            # Σ = R S Sᵀ Rᵀ. Compute Σ⁻¹ for evaluation.
            sx = gaussians.scale[:, 0].clamp(min=1e-6)
            sy = gaussians.scale[:, 1].clamp(min=1e-6)
            # Σ⁻¹ = R diag(1/sx², 1/sy²) Rᵀ.
            inv_sx2 = 1.0 / (sx * sx)
            inv_sy2 = 1.0 / (sy * sy)
            # 2x2 inverse cov per Gaussian
            a = cos_t * cos_t * inv_sx2 + sin_t * sin_t * inv_sy2  # (N,)
            b = cos_t * sin_t * (inv_sx2 - inv_sy2)
            d = sin_t * sin_t * inv_sx2 + cos_t * cos_t * inv_sy2
        else:
            a = conic[:, 0]
            b = conic[:, 1]
            d = conic[:, 2]

        out = torch.zeros((h, w, gaussians.feat_dim), device=device, dtype=gaussians.feat.dtype)
        # Loop Gaussian-major to keep memory bounded for arbitrary N.
        for i in range(n):
            dx = grid_x - gaussians.xy[i, 0]  # (H, W)
            dy = grid_y - gaussians.xy[i, 1]
            # Quadratic form: dx² a + 2 dx dy b + dy² d
            quad = dx * dx * a[i] + 2.0 * dx * dy * b[i] + dy * dy * d[i]
            weight = torch.exp(-0.5 * quad).unsqueeze(-1)  # (H, W, 1)
            out = out + weight * gaussians.feat[i]
        return out.permute(2, 0, 1).contiguous()


__all__ = [
    "GaussianBatch",
    "Rasterizer",
    "TILE_SIZE",
]

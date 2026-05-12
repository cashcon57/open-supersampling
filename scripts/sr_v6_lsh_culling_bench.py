"""LSH-vs-AABB Gaussian-culling micro-bench, no trained model.

Replicates the per-tile culling test from "N-Dimensional Gaussians for
Fitting of High Dimensional Functions" (Diolatzis et al. 2024,
https://arxiv.org/abs/2405.20067) at 2D and on the per-Gaussian sizes
the OSS Gaussian canvas actually carries (pico-tier ~2K, standard ~5K,
heavy ~15K).

For each tile in an HR image, we compute the set of Gaussians whose
3 sigma footprint reaches that tile by two methods:

  (A) Screen-extent / AABB:  project each Gaussian's principal-axis
      bounding box onto the tile rect, keep if overlap.  This is what
      gsplat-style tile binning does today.

  (B) LSH-projection:  for k random unit vectors r, project both the
      tile center q and the Gaussian mean m onto r, keep if
      | q . r - m . r | <= 3 sigma_r  where sigma_r^2 = r . V . r.
      Repeat for every vector; survive if ALL projections pass.

Reports for each Gaussian count and (B)'s k:
  - mean # Gaussians per tile that survive (lower = cheaper render)
  - LSH culling time (ms)
  - AABB culling time (ms)
  - LSH vs AABB reduction factor

If LSH cuts substantially more Gaussians per tile than AABB at
comparable preprocessing cost, integrating LSH binning into the OSS
rasterizer is worth the work.  If it's similar or worse in 2D, the
paper's speedup is dimensionality-dependent and the 2D case doesn't
benefit.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def _synthetic_canvas(n_gaussians: int, hw: tuple[int, int], seed: int = 42, device: str = "cpu"):
    """Build a synthetic 2D Gaussian set distributed roughly uniformly
    over an hw image, with anisotropic scales (some elongated, some
    round)."""
    g = torch.Generator(device=device).manual_seed(seed)
    h, w = hw
    xy = torch.rand((n_gaussians, 2), generator=g, device=device)
    xy[:, 0] *= float(w)
    xy[:, 1] *= float(h)
    # Mean scale ~3-12 px (covers typical pico-tier canvas)
    s = torch.empty((n_gaussians, 2), device=device)
    s[:, 0] = 3.0 + 9.0 * torch.rand((n_gaussians,), generator=g, device=device)
    s[:, 1] = 3.0 + 9.0 * torch.rand((n_gaussians,), generator=g, device=device)
    rot = (math.pi * 2) * torch.rand((n_gaussians,), generator=g, device=device)
    return xy, s, rot


def _covariance_from_scales_rotations(scales: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    n = scales.shape[0]
    cos = torch.cos(rot)
    sin = torch.sin(rot)
    R = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )  # (N, 2, 2)
    S = torch.diag_embed(scales.clamp(min=0.0).square())  # (N, 2, 2)
    return R @ S @ R.transpose(-1, -2)  # (N, 2, 2)


def aabb_cull(
    xy: torch.Tensor, scales: torch.Tensor, rot: torch.Tensor,
    tile_centers: torch.Tensor, tile_size: float,
) -> torch.Tensor:
    """For each tile, return a boolean mask of Gaussians whose 3 sigma
    AABB overlaps the tile rectangle. tile_centers is (T, 2).
    Returns (T, N) bool tensor."""
    # Principal axis bounds: project each Gaussian to its principal
    # axes, the 3 sigma ellipse fits in an OBB but the screen-aligned
    # AABB is principal_axis_max_scale * 3.
    radius = 3.0 * scales.max(dim=-1).values  # (N,)
    half = tile_size * 0.5
    # Broadcasting: tile center (T,1,2) vs Gaussian xy (1,N,2)
    diff = tile_centers.unsqueeze(1) - xy.unsqueeze(0)  # (T, N, 2)
    abs_diff = diff.abs()
    # Overlap if both axes within radius + half
    overlap = (abs_diff[..., 0] <= radius.unsqueeze(0) + half) & \
              (abs_diff[..., 1] <= radius.unsqueeze(0) + half)
    return overlap


def lsh_cull(
    xy: torch.Tensor, scales: torch.Tensor, rot: torch.Tensor,
    tile_centers: torch.Tensor, k: int, seed: int = 0,
    tile_size: float = 16.0,
) -> torch.Tensor:
    """For each tile, return a boolean mask of Gaussians passing all k
    LSH-projection 3 sigma tests."""
    n = xy.shape[0]
    t_count = tile_centers.shape[0]
    V = _covariance_from_scales_rotations(scales, rot)  # (N, 2, 2)
    g_lsh = torch.Generator(device=xy.device).manual_seed(seed)
    # Random unit vectors r: (k, 2)
    r_raw = torch.randn((k, 2), generator=g_lsh, device=xy.device)
    r = r_raw / r_raw.norm(dim=-1, keepdim=True)
    # Project tile centers and gaussian means to each r:
    #   q_r = q @ r.T -> (T, k)
    #   m_r = m @ r.T -> (N, k)
    q_r = tile_centers @ r.t()  # (T, k)
    m_r = xy @ r.t()             # (N, k)
    # sigma_r^2 = r^T V r, broadcast across r: (N, k)
    # sigma_r^2[n, j] = r[j]^T V[n] r[j]
    Vr = torch.einsum("nij,kj->nki", V, r)         # (N, k, 2)
    sigma_sq = torch.einsum("nki,ki->nk", Vr, r)   # (N, k)
    sigma = sigma_sq.clamp(min=1e-12).sqrt()        # (N, k)
    # |q_r - m_r| <= 3 sigma_r + tile_half
    half = 0.5 * tile_size
    diff = (q_r.unsqueeze(1) - m_r.unsqueeze(0)).abs()  # (T, N, k)
    sigma_b = sigma.unsqueeze(0)                         # (1, N, k)
    pass_per_dim = diff <= (3.0 * sigma_b + half)
    return pass_per_dim.all(dim=-1)  # (T, N)


def bench(n_gaussians: int, hw: tuple[int, int], tile_size: int, k_values: list[int], device: str = "cpu") -> dict:
    h, w = hw
    xy, scales, rot = _synthetic_canvas(n_gaussians, hw, device=device)
    # Tile centers: every tile_size px
    ty = torch.arange(tile_size / 2, h, tile_size, device=device)
    tx = torch.arange(tile_size / 2, w, tile_size, device=device)
    grid_y, grid_x = torch.meshgrid(ty, tx, indexing="ij")
    tile_centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    n_tiles = tile_centers.shape[0]

    # AABB
    t0 = time.perf_counter()
    aabb_mask = aabb_cull(xy, scales, rot, tile_centers, float(tile_size))
    if device == "cuda":
        torch.cuda.synchronize()
    aabb_ms = (time.perf_counter() - t0) * 1000.0
    aabb_per_tile = aabb_mask.sum(dim=1).float().mean().item()

    out_k = {}
    for k in k_values:
        t0 = time.perf_counter()
        lsh_mask = lsh_cull(xy, scales, rot, tile_centers, k=k, tile_size=float(tile_size))
        if device == "cuda":
            torch.cuda.synchronize()
        lsh_ms = (time.perf_counter() - t0) * 1000.0
        lsh_per_tile = lsh_mask.sum(dim=1).float().mean().item()
        # Did LSH agree with AABB? (LSH should be a SUBSET of AABB
        # because AABB is a conservative outer bound on what could
        # touch a tile; LSH being tighter means it culls more.)
        agreement = (lsh_mask & aabb_mask).sum().item() / max(aabb_mask.sum().item(), 1)
        out_k[k] = {
            "lsh_ms": lsh_ms,
            "lsh_mean_gaussians_per_tile": lsh_per_tile,
            "reduction_vs_aabb": (aabb_per_tile / max(lsh_per_tile, 1e-9)),
            "lsh_subset_of_aabb_fraction": agreement,
        }

    return {
        "n_gaussians": n_gaussians,
        "image_hw": list(hw),
        "tile_size": tile_size,
        "n_tiles": n_tiles,
        "device": device,
        "aabb": {
            "ms": aabb_ms,
            "mean_gaussians_per_tile": aabb_per_tile,
        },
        "lsh_by_k": out_k,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lsh_bench.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gaussian-counts", default="500,2000,5000,15000")
    parser.add_argument("--k-values", default="1,2,4,8,16",
                        help="Number of random projection vectors to test per LSH.")
    parser.add_argument("--hw", default="480,640", help="Image (H,W).")
    parser.add_argument("--tile-size", default=16, type=int)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU.")
        args.device = "cpu"

    hw = tuple(int(x) for x in args.hw.split(","))
    g_counts = [int(s.strip()) for s in args.gaussian_counts.split(",")]
    k_values = [int(s.strip()) for s in args.k_values.split(",")]

    all_results = []
    for n in g_counts:
        r = bench(n, hw, args.tile_size, k_values, args.device)
        all_results.append(r)
        print(f"\n=== N={n} Gaussians, tile={args.tile_size}px, hw={hw} ===")
        print(f"  AABB:  {r['aabb']['ms']:.2f} ms,  mean {r['aabb']['mean_gaussians_per_tile']:.1f} Gauss/tile")
        for k, lsh in r["lsh_by_k"].items():
            print(
                f"  LSH k={k:2d}: {lsh['lsh_ms']:.2f} ms,  "
                f"mean {lsh['lsh_mean_gaussians_per_tile']:.1f} Gauss/tile,  "
                f"reduction {lsh['reduction_vs_aabb']:.2f}x  "
                f"(subset-of-AABB {lsh['lsh_subset_of_aabb_fraction']*100:.1f}%)"
            )

    args.output.write_text(json.dumps({"results": all_results}, indent=2))
    print(f"\n[bench] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

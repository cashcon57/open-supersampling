#!/usr/bin/env python3
"""Technique I: tile-mask cache-hit estimate from synthetic or ckpt canvas drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from f_topk_stats import _load_model


def aabb(xy: torch.Tensor, scale: torch.Tensor, tile_size: int, h: int, w: int) -> torch.Tensor:
    radius = 3.0 * scale.amax(dim=-1)
    ntx = (w + tile_size - 1) // tile_size
    nty = (h + tile_size - 1) // tile_size
    tx0 = torch.floor((xy[:, 0] - radius) / tile_size).clamp(0, ntx).to(torch.int32)
    tx1 = torch.ceil((xy[:, 0] + radius) / tile_size).clamp(0, ntx).to(torch.int32)
    ty0 = torch.floor((xy[:, 1] - radius) / tile_size).clamp(0, nty).to(torch.int32)
    ty1 = torch.ceil((xy[:, 1] + radius) / tile_size).clamp(0, nty).to(torch.int32)
    return torch.stack([tx0, ty0, tx1, ty1], dim=1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(2)
    device = torch.device(args.device)
    model = _load_model(args.ckpt, device)
    x = torch.rand(1, model.cfg.in_channels, args.height, args.width, device=device)
    feats = model.activation(model.pixel_head(model.backbone(x)))
    spawned = model.gaussian_spawner(feats)
    xy0 = spawned.positions.reshape(-1, 2).float()
    scale0 = spawned.scales.reshape(-1, 2).float().clamp_min(1.0e-3)
    h, w = args.height * model.scale, args.width * model.scale
    prev = aabb(xy0, scale0, 16, h, w)
    stable = []
    for t in range(1, args.frames):
        # Synthetic fallback drift: 0.25 px/frame Gaussian center noise and
        # 0.5% scale noise. Real-frame mode should replace this by captured
        # consecutive spawned states from --input-dir.
        xy = xy0 + torch.randn_like(xy0) * (0.25 * t)
        scale = scale0 * (1.0 + torch.randn_like(scale0) * (0.005 * t))
        cur = aabb(xy, scale.clamp_min(1.0e-3), 16, h, w)
        stable.append(torch.all(cur == prev, dim=1).float().mean().item())
        prev = cur
    print(json.dumps({
        "technique": "I",
        "mode": "ckpt" if args.ckpt and args.ckpt.exists() else "synthetic_fallback",
        "ckpt": None if args.ckpt is None else str(args.ckpt),
        "gaussians": int(xy0.shape[0]),
        "frames": args.frames,
        "same_tile_aabb_fraction_per_transition": stable,
        "mean_cache_hit_estimate": float(np.mean(stable)) if stable else 1.0,
        "verdict": "cache if measured real-frame hit rate stays high; synthetic value is only a smoke check",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

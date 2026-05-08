#!/usr/bin/env python3
"""Technique G: Sobel edge-tile fraction on v6 LR feature maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from f_topk_stats import _load_model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--threshold-quantile", type=float, default=0.70)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("docs/coordination/phase4-elegance-artifacts/g_edge_tiles_hist.png"))
    args = ap.parse_args()

    torch.manual_seed(1)
    device = torch.device(args.device)
    model = _load_model(args.ckpt, device)
    x = torch.rand(1, model.cfg.in_channels, args.height, args.width, device=device)
    feats = model.backbone(x).float()
    gray = feats.square().mean(dim=1, keepdim=True).sqrt()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)
    tile = int(model.cfg.tile_size_lr)
    pooled = F.avg_pool2d(mag, kernel_size=tile, stride=tile).reshape(-1)
    arr = pooled.cpu().numpy()
    threshold = float(np.quantile(arr, args.threshold_quantile))
    edge = arr > threshold

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=32)
        plt.axvline(threshold)
        plt.xlabel("mean Sobel magnitude per LR tile")
        plt.ylabel("tiles")
        plt.tight_layout()
        plt.savefig(args.out)
        plt.close()
    except Exception:
        args.out = Path("")

    print(json.dumps({
        "technique": "G",
        "mode": "ckpt" if args.ckpt and args.ckpt.exists() else "synthetic_fallback",
        "ckpt": None if args.ckpt is None else str(args.ckpt),
        "tiles": int(arr.size),
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "edge_fraction": float(np.mean(edge)),
        "flat_fraction": float(1.0 - np.mean(edge)),
        "histogram_png": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

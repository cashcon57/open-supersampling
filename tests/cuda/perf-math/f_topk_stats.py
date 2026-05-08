#!/usr/bin/env python3
"""Technique F: per-pixel K needed for 99% Gaussian weight mass.

Runs against a v6 checkpoint when --ckpt is provided. Without a checkpoint, it
uses the untrained v6.1-pico-shaped model as a deterministic smoke fixture and
marks the output as synthetic_fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_model(ckpt: Path | None, device: torch.device):
    from oss.sr.v6.model import V6Config, V6Model

    cfg = V6Config(backbone="hat-tiny", in_channels=9, canvas_capacity=4096)
    state = None
    if ckpt is not None and ckpt.exists():
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg_raw = raw.get("v6_config") or raw.get("config")
        if isinstance(cfg_raw, dict):
            cfg = V6Config(**{**cfg.__dict__, **cfg_raw})
        state = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict")
    model = V6Model(cfg).to(device).eval()
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("docs/coordination/phase4-elegance-artifacts/f_topk_hist.png"))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    model = _load_model(args.ckpt, device)
    x = torch.rand(1, model.cfg.in_channels, args.height, args.width, device=device)
    feats = model.activation(model.pixel_head(model.backbone(x)))
    spawned = model.gaussian_spawner(feats)
    xy = spawned.positions.reshape(-1, 2)[0:512].float()
    scale = spawned.scales.reshape(-1, 2)[0:512].float().clamp_min(1.0e-3)
    rot = spawned.rotations.reshape(-1)[0:512].float()

    h, w = args.height * model.scale, args.width * model.scale
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cos_t = torch.cos(rot)
    sin_t = torch.sin(rot)
    inv_sx2 = 1.0 / (scale[:, 0] * scale[:, 0])
    inv_sy2 = 1.0 / (scale[:, 1] * scale[:, 1])
    a = cos_t * cos_t * inv_sx2 + sin_t * sin_t * inv_sy2
    b = cos_t * sin_t * (inv_sx2 - inv_sy2)
    d = sin_t * sin_t * inv_sx2 + cos_t * cos_t * inv_sy2
    weights = []
    for i in range(xy.shape[0]):
        dx = xx - xy[i, 0]
        dy = yy - xy[i, 1]
        q = a[i] * dx * dx + 2.0 * b[i] * dx * dy + d[i] * dy * dy
        weights.append(torch.exp(-0.5 * q).reshape(-1))
    wmat = torch.stack(weights, dim=1)
    sorted_w = torch.sort(wmat, dim=1, descending=True).values
    csum = torch.cumsum(sorted_w, dim=1)
    total = csum[:, -1:].clamp_min(1.0e-12)
    k99 = (csum < 0.99 * total).sum(dim=1) + 1
    arr = k99.cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=np.arange(1, int(arr.max()) + 2))
        plt.xlabel("K for 99% weight mass")
        plt.ylabel("pixels")
        plt.tight_layout()
        plt.savefig(args.out)
        plt.close()
    except Exception:
        args.out = Path("")

    print(json.dumps({
        "technique": "F",
        "mode": "ckpt" if args.ckpt and args.ckpt.exists() else "synthetic_fallback",
        "ckpt": None if args.ckpt is None else str(args.ckpt),
        "pixels": int(arr.size),
        "gaussians_considered": int(xy.shape[0]),
        "k99_p50": float(np.percentile(arr, 50)),
        "k99_p95": float(np.percentile(arr, 95)),
        "k99_p99": float(np.percentile(arr, 99)),
        "fraction_k_le_4": float(np.mean(arr <= 4)),
        "fraction_k_le_8": float(np.mean(arr <= 8)),
        "histogram_png": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

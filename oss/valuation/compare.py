"""End-to-end comparison: ORD-only vs Paired vs OIDN baseline.

Outputs CSV: model, scene, psnr, ssim, lpips, mean_ms.
"""
from __future__ import annotations
import argparse
import csv
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from oss.valuation.bench import bench_model
from oss.valuation.metrics import psnr, ssim, lpips_dist
from oss.model import ORD, ORU
from oss.model.adapter import PairedORS
from oss.train.data import ORSDataset

log = logging.getLogger(__name__)


def _run_oidn(noisy, albedo, normal):
    """Try OIDN Apache-2.0 baseline. Returns denoised tensor or None if unavailable."""
    try:
        import oidn  # noqa: F401
    except ImportError:
        return None
    raise NotImplementedError("Wire OIDN binding once selected (v0.2).")


def _ldr(x):
    return (x / (x + 1.0)).clamp(0, 1)


def _eval_quality(pred, gt):
    pred_ldr = _ldr(pred)
    gt_ldr = _ldr(gt)
    return {
        "psnr": psnr(pred, gt).item(),
        "ssim": ssim(pred_ldr, gt_ldr).item(),
        "lpips": lpips_dist(pred_ldr * 2 - 1, gt_ldr * 2 - 1).item(),
    }


def _smoke_dataset():
    H, W = 64, 64
    return [{
        "noisy":        torch.randn(1, 3, H, W),
        "ground_truth": torch.randn(1, 3, H, W),
        "aux":          torch.randn(1, 11, H, W),
        "history":      torch.randn(1, 3, H, W),
        "depth":        torch.randn(1, 1, H, W),
        "motion":       torch.randn(1, 2, H, W),
        "albedo":       torch.randn(1, 3, H, W),
        "normal":       torch.randn(1, 3, H, W),
    }]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ord-ckpt", default="results/ord/ord.pth")
    p.add_argument("--oru-ckpt", default="results/oru/oru.pth")
    p.add_argument("--paired-ckpt", default="results/paired/paired.pth")
    p.add_argument("--data", default="data/bistro_mvp")
    p.add_argument("--out", default="results/comparison.csv")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        samples = _smoke_dataset()
    else:
        ds = ORSDataset(root=Path(args.data), augment=False)
        samples = [{k: v.unsqueeze(0).to(device) for k, v in ds[i].items()} for i in range(len(ds))]

    rows = []

    # ORD-only
    if Path(args.ord_ckpt).exists() or args.smoke_test:
        ord_model = ORD(tier="standard").to(device).train(False)
        if Path(args.ord_ckpt).exists():
            ord_model.load_state_dict(torch.load(args.ord_ckpt, map_location=device)["model"])
        with torch.no_grad():
            for i, s in enumerate(samples):
                pred, _ = ord_model(s["noisy"], s["aux"], s["history"])
                row = {"model": "ord", "scene": i, **_eval_quality(pred, s["ground_truth"])}
                row["mean_ms"] = bench_model(lambda: ord_model(s["noisy"], s["aux"], s["history"]))["mean_ms"]
                rows.append(row)

    # Paired
    if Path(args.paired_ckpt).exists() or args.smoke_test:
        ord_model = ORD(tier="standard").to(device).train(False)
        oru_model = ORU(input_mode="features", scale_factor=2.0, tier="standard").to(device).train(False)
        if Path(args.paired_ckpt).exists():
            sd = torch.load(args.paired_ckpt, map_location=device)
            ord_model.load_state_dict(sd["ord"])
            oru_model.load_state_dict(sd["oru"])
        pair = PairedORS(ord_model, oru_model)
        with torch.no_grad():
            for i, s in enumerate(samples):
                ds_kw = dict(scale_factor=0.5, mode="bilinear", align_corners=False)
                noisy_lr  = F.interpolate(s["noisy"], **ds_kw)
                aux_lr    = F.interpolate(s["aux"], **ds_kw)
                hist_lr   = F.interpolate(s["history"], **ds_kw)
                depth_lr  = F.interpolate(s["depth"], **ds_kw)
                motion_lr = F.interpolate(s["motion"], **ds_kw)
                _, pred_hi = pair(noisy=noisy_lr, aux=aux_lr, history=hist_lr,
                                  depth=depth_lr, motion=motion_lr)
                row = {"model": "paired", "scene": i, **_eval_quality(pred_hi, s["ground_truth"])}
                row["mean_ms"] = bench_model(lambda: pair(
                    noisy=noisy_lr, aux=aux_lr, history=hist_lr,
                    depth=depth_lr, motion=motion_lr))["mean_ms"]
                rows.append(row)

    # OIDN baseline (stub for v0.1)
    with torch.no_grad():
        for i, s in enumerate(samples):
            denoised = _run_oidn(s["noisy"], s["albedo"], s["normal"])
            if denoised is None:
                log.info("OIDN unavailable; recording NaN row for scene %d", i)
                rows.append({"model": "oidn", "scene": i,
                             "psnr": float("nan"), "ssim": float("nan"),
                             "lpips": float("nan"), "mean_ms": float("nan")})
                continue
            row = {"model": "oidn", "scene": i, **_eval_quality(denoised, s["ground_truth"])}
            rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scene", "psnr", "ssim", "lpips", "mean_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

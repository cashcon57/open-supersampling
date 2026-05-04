"""Fixed-batch A/B between two SR-CNN checkpoints.

Scores BOTH checkpoints on the SAME deterministic batch of held-out frames
and reports PSNR + LPIPS deltas. Used for v3 (L1+SSIM) vs v4 (L1+SSIM+LPIPS)
comparison so we know whether v4 is a real improvement before building
temporal extensions on top of it.

Usage:
    python scripts/sr_v3_vs_v4_ab.py \\
        --ckpt-a <train-host-data>\\checkpoints\\srcnn-prod-v3\\step-00310000.pt \\
        --ckpt-b <train-host-data>\\checkpoints\\srcnn-prod-v4-lpips\\step-00385000.pt \\
        --eval-scene CitySample \\
        --dataset-root <train-host-data>\\datasets\\srgd \\
        --n-samples 64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oss.gaussian.data import EngineAliasedLRSynth, SRGDGaussianDataset, collate_examples
from oss.sr import build_sr_model


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(F.mse_loss(pred.float(), target.float()).item())
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


def _load_model(ckpt_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "standard")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"
    model = build_sr_model(
        model_kind=factory_kind, tier=tier, in_channels=12, scale=2,
    ).to(device)
    model.load_state_dict(ckpt["sr_model"])
    model.train(False)
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-a", type=Path, required=True, help="First checkpoint (e.g. v3)")
    p.add_argument("--ckpt-b", type=Path, required=True, help="Second checkpoint (e.g. v4)")
    p.add_argument("--eval-scene", type=str, default="CitySample")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device
    torch.manual_seed(args.seed)

    # Load both models.
    print(f"Loading A: {args.ckpt_a}")
    model_a = _load_model(args.ckpt_a, device)
    print(f"Loading B: {args.ckpt_b}")
    model_b = _load_model(args.ckpt_b, device)

    # LPIPS — single instance shared between A and B.
    lpips_fn = None
    try:
        import lpips  # type: ignore[import-not-found]
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.train(False)
    except Exception as e:
        print(f"WARN: LPIPS unavailable ({e}) - PSNR only")

    # Deterministic dataset (same frames every run).
    lr_synth = EngineAliasedLRSynth(
        enable_jitter=True, enable_taa_blur=True, enable_jpeg=True,
        jpeg_quality=85, blur_sigma=1.5,
    )
    candidates = [args.dataset_root, args.dataset_root / "srgd"]
    srgd_root = next(
        (c for c in candidates if (c / "data" / "GameEngineData").is_dir()),
        None,
    )
    if srgd_root is None:
        print(f"FAIL: no SRGD dataset under {candidates}")
        return 1
    ds = SRGDGaussianDataset(
        root=srgd_root, scale=2.0, lr_synth=lr_synth, scene=args.eval_scene,
        force_synth_lr=True,
    )
    print(f"eval_scene={args.eval_scene}  n_frames={len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=collate_examples, drop_last=True,
    )

    psnr_a: list[float] = []
    psnr_b: list[float] = []
    psnr_bic: list[float] = []
    lpips_a: list[float] = []
    lpips_b: list[float] = []
    lpips_bic: list[float] = []

    def _lpips(pred: torch.Tensor, target: torch.Tensor) -> float | None:
        if lpips_fn is None:
            return None
        p = pred.unsqueeze(0) * 2.0 - 1.0
        t = target.unsqueeze(0) * 2.0 - 1.0
        return float(lpips_fn(p, t).item())

    with torch.no_grad():
        for batch in loader:
            if len(psnr_a) >= args.n_samples:
                break
            lr = batch["lr_frame"].to(device)
            depth = batch["depth"].to(device)
            motion = batch["motion"].to(device)
            normals = batch["normals"].to(device)
            canvas = batch["canvas_hint"].to(device)
            gt_hr = batch["gt_hr_frame"].to(device)

            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            H_hr, W_hr = gt_hr.shape[-2:]
            bicubic_hr = F.interpolate(
                lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)
            out_a = model_a(x).clamp(0.0, 1.0)
            out_b = model_b(x).clamp(0.0, 1.0)

            for b_idx in range(lr.shape[0]):
                if len(psnr_a) >= args.n_samples:
                    break
                psnr_a.append(_psnr(out_a[b_idx], gt_hr[b_idx]))
                psnr_b.append(_psnr(out_b[b_idx], gt_hr[b_idx]))
                psnr_bic.append(_psnr(bicubic_hr[b_idx], gt_hr[b_idx]))
                la = _lpips(out_a[b_idx], gt_hr[b_idx])
                lb = _lpips(out_b[b_idx], gt_hr[b_idx])
                lc = _lpips(bicubic_hr[b_idx], gt_hr[b_idx])
                if la is not None:
                    lpips_a.append(la)
                    lpips_b.append(lb)
                    lpips_bic.append(lc)

    n = len(psnr_a)
    if n == 0:
        print("FAIL: no samples evaluated")
        return 1

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    print()
    print(f"=== A/B fixed-batch eval ({args.eval_scene}, n={n}) ===")
    print(f"  ckpt_a = {args.ckpt_a.name}")
    print(f"  ckpt_b = {args.ckpt_b.name}")
    print()
    print(f"PSNR (dB, higher is better)")
    print(f"  A          : {_mean(psnr_a):6.3f}")
    print(f"  B          : {_mean(psnr_b):6.3f}")
    print(f"  bicubic    : {_mean(psnr_bic):6.3f}")
    print(f"  B-vs-A     : {_mean(psnr_b)-_mean(psnr_a):+6.3f} dB")
    print(f"  A>bicubic  : {sum(1 for a,c in zip(psnr_a,psnr_bic) if a>c)}/{n}")
    print(f"  B>bicubic  : {sum(1 for b,c in zip(psnr_b,psnr_bic) if b>c)}/{n}")
    print(f"  B>A        : {sum(1 for a,b in zip(psnr_a,psnr_b) if b>a)}/{n}")
    if lpips_a:
        a_mean = _mean(lpips_a)
        b_mean = _mean(lpips_b)
        rel_pct = 100.0 * (b_mean - a_mean) / a_mean if a_mean else 0.0
        print()
        print(f"LPIPS-VGG (lower is better)")
        print(f"  A          : {a_mean:6.4f}")
        print(f"  B          : {b_mean:6.4f}")
        print(f"  bicubic    : {_mean(lpips_bic):6.4f}")
        print(f"  B-vs-A     : {b_mean - a_mean:+7.4f}  ({rel_pct:+5.1f}%)")
        print(f"  A<bicubic  : {sum(1 for a,c in zip(lpips_a,lpips_bic) if a<c)}/{n}")
        print(f"  B<bicubic  : {sum(1 for b,c in zip(lpips_b,lpips_bic) if b<c)}/{n}")
        print(f"  B<A        : {sum(1 for a,b in zip(lpips_a,lpips_b) if b<a)}/{n}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

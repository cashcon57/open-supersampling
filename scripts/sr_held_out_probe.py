"""Held-out scene generalisation probe for OSS-SR (CNN backbone).

Loads an SR-CNN checkpoint trained on one SRGD scene and scores against
bicubic on a different SRGD scene. Same idea as
`held_out_scene_probe.py` but for the SR-track checkpoint format
(saved under `sr_model` key by the trainer).

Usage:
    python scripts/sr_held_out_probe.py \\
        --checkpoint <train-host-data>\\checkpoints\\sprint4-srcnn-fixed\\step-00001000.pt \\
        --eval-scene CitySample \\
        --dataset-root <train-host-data>\\datasets\\srgd
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-scene", type=str, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "lite")
    model_kind = saved_args.get("model_kind", "sr_cnn")
    sr_backbone = saved_args.get("sr_backbone", "simple")

    print(f"Loaded: {args.checkpoint}")
    print(f"  tier={tier}  model_kind={model_kind}  sr_backbone={sr_backbone}")

    # Trainer's checkpoint stores model_kind ∈ {gaussian, sr_cnn, sr_rrdb} and
    # sr_backbone ∈ {simple, rrdb}. Factory takes model_kind ∈ {simple, rrdb}.
    factory_kind = "rrdb" if (model_kind == "sr_rrdb" or sr_backbone == "rrdb") else "simple"
    sr_model = build_sr_model(
        model_kind=factory_kind,
        tier=tier,
        in_channels=12,
        scale=2,
    ).to(device)
    sr_model.load_state_dict(ckpt["sr_model"])
    sr_model.train(False)

    lr_synth = EngineAliasedLRSynth(
        enable_jitter=True,
        enable_taa_blur=True,
        enable_jpeg=True,
        jpeg_quality=85,
        blur_sigma=1.5,
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
    print(f"  eval_scene={args.eval_scene}  n_frames={len(ds)}")
    if len(ds) == 0:
        print(f"FAIL: scene {args.eval_scene!r} is empty")
        return 1

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=collate_examples, drop_last=True,
    )

    model_psnrs: list[float] = []
    bicubic_psnrs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            if len(model_psnrs) >= args.n_samples:
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

            out = sr_model(x).clamp(0.0, 1.0)

            for b_idx in range(lr.shape[0]):
                if len(model_psnrs) >= args.n_samples:
                    break
                model_psnrs.append(_psnr(out[b_idx], gt_hr[b_idx]))
                bicubic_psnrs.append(_psnr(bicubic_hr[b_idx], gt_hr[b_idx]))

    n = len(model_psnrs)
    if n == 0:
        print("FAIL: no samples evaluated")
        return 1

    model_mean = sum(model_psnrs) / n
    bicubic_mean = sum(bicubic_psnrs) / n
    beats = sum(1 for m, b in zip(model_psnrs, bicubic_psnrs) if m > b)

    print()
    print(f"=== SR-CNN held-out probe ({args.eval_scene}) ===")
    print(f"n_samples         = {n}")
    print(f"model_psnr_mean   = {model_mean:.2f} dB")
    print(f"bicubic_psnr_mean = {bicubic_mean:.2f} dB")
    print(f"margin            = {model_mean - bicubic_mean:+.2f} dB")
    print(f"beats_bicubic     = {beats}/{n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

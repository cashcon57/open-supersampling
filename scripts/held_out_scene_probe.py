"""Held-out scene generalisation probe.

Loads a checkpoint trained on one SRGD scene and scores against bicubic on a
different SRGD scene. Tells us whether V0.5 (Gaussian splat + pixel-residual
head) is genuinely learning game-engine SR or just memorising the training
scene.

Usage:
    python scripts/held_out_scene_probe.py \
        --checkpoint <train-host-data>\\checkpoints\\sprint4-v05\\step-00001000.pt \
        --eval-scene CSGO \
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
from oss.gaussian.network import (
    CovariancePriorBank,
    OutputHead,
    PixelResidualHead,
    param_net_for_tier,
)
from oss.gaussian.renderer import Rasterizer


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
    p.add_argument("--enable-pixel-residual", action="store_true", default=True)
    p.add_argument("--no-pixel-residual", dest="enable_pixel_residual", action="store_false")
    args = p.parse_args()

    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "lite")
    bank_size = saved_args.get("bank_size", 16)
    enable_gbuffer_bias = saved_args.get("enable_gbuffer_bias", True)
    pixel_residual_hidden = saved_args.get("pixel_residual_hidden", 32)
    enable_pixel_residual = args.enable_pixel_residual and saved_args.get(
        "enable_pixel_residual", True
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  tier={tier}  bank_size={bank_size}")
    print(f"  enable_gbuffer_bias={enable_gbuffer_bias}")
    print(f"  enable_pixel_residual={enable_pixel_residual}")

    bank = CovariancePriorBank(learnable=False).to(device)
    net = param_net_for_tier(tier, bank_size=bank_size).to(device)
    head = OutputHead(
        bank=bank, k_per_tile=net.k_per_tile, enable_gbuffer_bias=enable_gbuffer_bias
    ).to(device)

    net.load_state_dict(ckpt["net"])
    bank.load_state_dict(ckpt["bank"])

    residual_head = None
    if enable_pixel_residual:
        residual_head = PixelResidualHead(
            in_channels=6, hidden_channels=pixel_residual_hidden
        ).to(device)
        if "residual_head" in ckpt:
            residual_head.load_state_dict(ckpt["residual_head"])
            print("  residual_head: loaded from checkpoint")
        else:
            print("  residual_head: NOT in checkpoint, using zero-init (splat-only output)")

    net.train(False)
    if residual_head is not None:
        residual_head.train(False)

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
        root=srgd_root,
        scale=2.0,
        lr_synth=lr_synth,
        scene=args.eval_scene,
        force_synth_lr=True,
    )
    print(f"  eval_scene={args.eval_scene}  n_frames={len(ds)}")

    if len(ds) == 0:
        print(f"FAIL: scene {args.eval_scene!r} is empty")
        return 1

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_examples,
        drop_last=True,
    )

    renderer = Rasterizer()

    model_psnrs: list[float] = []
    bicubic_psnrs: list[float] = []

    tile = net.tile_size
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

            scale_int = int(round(gt_hr.shape[-2] / lr.shape[-2]))
            lr_h, lr_w = lr.shape[-2:]
            lr_h_a = (lr_h // tile) * tile
            lr_w_a = (lr_w // tile) * tile
            if (lr_h_a, lr_w_a) != (lr_h, lr_w):
                top = (lr_h - lr_h_a) // 2
                left = (lr_w - lr_w_a) // 2
                lr = lr[..., top : top + lr_h_a, left : left + lr_w_a]
                depth = depth[..., top : top + lr_h_a, left : left + lr_w_a]
                motion = motion[..., top : top + lr_h_a, left : left + lr_w_a]
                normals = normals[..., top : top + lr_h_a, left : left + lr_w_a]
                canvas = canvas[..., top : top + lr_h_a, left : left + lr_w_a]
                hr_top = top * scale_int
                hr_left = left * scale_int
                gt_hr = gt_hr[
                    ...,
                    hr_top : hr_top + lr_h_a * scale_int,
                    hr_left : hr_left + lr_w_a * scale_int,
                ]

            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            H_hr, W_hr = gt_hr.shape[-2:]

            bicubic_hr = F.interpolate(
                lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)

            raw = net(x)
            for b_idx in range(lr.shape[0]):
                if len(model_psnrs) >= args.n_samples:
                    break
                gaussians = head.to_gaussian_batch(
                    raw,
                    batch_index=b_idx,
                    depth=depth[b_idx : b_idx + 1],
                    normals=normals[b_idx : b_idx + 1],
                )
                rendered = renderer(gaussians, output_hw=(H_hr, W_hr)).clamp(0.0, 1.0)
                if residual_head is not None:
                    res = residual_head(
                        rendered.unsqueeze(0), bicubic_hr[b_idx : b_idx + 1]
                    ).squeeze(0)
                    rendered = (rendered + res).clamp(0.0, 1.0)

                model_psnrs.append(_psnr(rendered, gt_hr[b_idx]))
                bicubic_psnrs.append(_psnr(bicubic_hr[b_idx], gt_hr[b_idx]))

    n = len(model_psnrs)
    if n == 0:
        print("FAIL: no samples evaluated")
        return 1

    model_mean = sum(model_psnrs) / n
    bicubic_mean = sum(bicubic_psnrs) / n
    beats = sum(1 for m, b in zip(model_psnrs, bicubic_psnrs) if m > b)

    print()
    print(f"=== Held-out-scene probe ({args.eval_scene}) ===")
    print(f"n_samples         = {n}")
    print(f"model_psnr_mean   = {model_mean:.2f} dB")
    print(f"bicubic_psnr_mean = {bicubic_mean:.2f} dB")
    print(f"margin            = {model_mean - bicubic_mean:+.2f} dB")
    print(f"beats_bicubic     = {beats}/{n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

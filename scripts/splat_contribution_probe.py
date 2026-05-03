"""Splat contribution probe — does the splat path contribute, or is the residual
CNN doing all the work?

Three modes scored against the same SRGD scene with the same checkpoint:
    --mode splat+residual   normal V0.5 path
    --mode splat-only       residual=0; output = clamp(splat_render)
    --mode zero+residual    splat_render = 0; residual head sees (zeros, lr_up)
    --mode bicubic+residual splat_render = bicubic_up; residual sees (lr_up, lr_up)

If splat+residual ≈ zero+residual ≈ bicubic+residual, the splats add no
information; the 12K residual CNN is the entire SR module.

Usage:
    python scripts/splat_contribution_probe.py \\
        --checkpoint <train-host-data>\\checkpoints\\sprint4-prod\\step-00060000.pt \\
        --eval-scene CitySample --dataset-root <train-host-data>\\datasets\\srgd \\
        --mode all
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


MODES = ("splat+residual", "splat-only", "zero+residual", "bicubic+residual")


def _run_mode(
    mode: str,
    *,
    net,
    head,
    bank,
    residual_head,
    renderer,
    loader,
    device,
    n_samples,
    tile,
):
    model_psnrs: list[float] = []
    bicubic_psnrs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            if len(model_psnrs) >= n_samples:
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
                if len(model_psnrs) >= n_samples:
                    break
                gaussians = head.to_gaussian_batch(
                    raw,
                    batch_index=b_idx,
                    depth=depth[b_idx : b_idx + 1],
                    normals=normals[b_idx : b_idx + 1],
                )
                splat = renderer(gaussians, output_hw=(H_hr, W_hr)).clamp(0.0, 1.0)

                if mode == "splat+residual":
                    res = residual_head(
                        splat.unsqueeze(0), bicubic_hr[b_idx : b_idx + 1]
                    ).squeeze(0)
                    out = (splat + res).clamp(0.0, 1.0)
                elif mode == "splat-only":
                    out = splat
                elif mode == "zero+residual":
                    zero_in = torch.zeros_like(splat).unsqueeze(0)
                    res = residual_head(
                        zero_in, bicubic_hr[b_idx : b_idx + 1]
                    ).squeeze(0)
                    # Without a splat term to add to, the residual IS the output.
                    out = res.clamp(0.0, 1.0)
                elif mode == "bicubic+residual":
                    bic_in = bicubic_hr[b_idx : b_idx + 1]
                    res = residual_head(bic_in, bic_in).squeeze(0)
                    out = (bic_in.squeeze(0) + res).clamp(0.0, 1.0)
                else:
                    raise ValueError(f"Unknown mode: {mode!r}")

                model_psnrs.append(_psnr(out, gt_hr[b_idx]))
                bicubic_psnrs.append(_psnr(bicubic_hr[b_idx], gt_hr[b_idx]))

    n = len(model_psnrs)
    if n == 0:
        return None
    model_mean = sum(model_psnrs) / n
    bicubic_mean = sum(bicubic_psnrs) / n
    beats = sum(1 for m, b in zip(model_psnrs, bicubic_psnrs) if m > b)
    return {
        "n": n,
        "model_mean": model_mean,
        "bicubic_mean": bicubic_mean,
        "margin": model_mean - bicubic_mean,
        "beats": beats,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-scene", type=str, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mode", choices=("all",) + MODES, default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "lite")
    bank_size = saved_args.get("bank_size", 16)
    enable_gbuffer_bias = saved_args.get("enable_gbuffer_bias", True)
    pixel_residual_hidden = saved_args.get("pixel_residual_hidden", 32)

    print(f"Loaded: {args.checkpoint}")
    print(f"  tier={tier}  bank_size={bank_size}  pixel_residual_hidden={pixel_residual_hidden}")

    bank = CovariancePriorBank(learnable=False).to(device)
    net = param_net_for_tier(tier, bank_size=bank_size).to(device)
    head = OutputHead(
        bank=bank, k_per_tile=net.k_per_tile, enable_gbuffer_bias=enable_gbuffer_bias
    ).to(device)
    net.load_state_dict(ckpt["net"])
    bank.load_state_dict(ckpt["bank"])

    residual_head = PixelResidualHead(in_channels=6, hidden_channels=pixel_residual_hidden).to(device)
    if "residual_head" in ckpt:
        residual_head.load_state_dict(ckpt["residual_head"])
        print("  residual_head: loaded from checkpoint")
    else:
        print("  residual_head: MISSING from checkpoint, using zero-init (this probe needs a real one)")

    net.train(False)
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
        root=srgd_root, scale=2.0, lr_synth=lr_synth, scene=args.eval_scene,
        force_synth_lr=True,
    )
    print(f"  eval_scene={args.eval_scene}  n_frames={len(ds)}")

    if len(ds) == 0:
        print(f"FAIL: scene {args.eval_scene!r} is empty")
        return 1

    renderer = Rasterizer()
    tile = net.tile_size

    modes_to_run = MODES if args.mode == "all" else (args.mode,)

    print()
    print(f"=== Splat contribution probe ({args.eval_scene}) ===")
    print(f"{'mode':22s}  {'n':>3s}  {'model':>7s}  {'bicubic':>8s}  {'margin':>8s}  beats")
    for mode in modes_to_run:
        # Re-seed loader so each mode sees the same shuffle order.
        torch.manual_seed(args.seed)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
            collate_fn=collate_examples, drop_last=True,
        )
        result = _run_mode(
            mode, net=net, head=head, bank=bank, residual_head=residual_head,
            renderer=renderer, loader=loader, device=device,
            n_samples=args.n_samples, tile=tile,
        )
        if result is None:
            print(f"{mode:22s}  --- no samples ---")
            continue
        print(
            f"{mode:22s}  {result['n']:3d}  "
            f"{result['model_mean']:7.2f}  {result['bicubic_mean']:8.2f}  "
            f"{result['margin']:+8.2f}  {result['beats']}/{result['n']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

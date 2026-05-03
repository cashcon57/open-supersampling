"""Quantify the quality cost of replacing bicubic-antialias skip with bilinear
in the ONNX export path.

The trained model uses ``F.interpolate(..., mode='bicubic', antialias=True)``
in its forward pass for the LR-bicubic skip. PyTorch 2.4.1 + opset 17 cannot
export the antialias variant to ONNX, so the deployed ONNX path falls back
to ``mode='bilinear', antialias=False``. This script measures the PSNR
delta on real held-out SRGD frames so we know the production quality cost.

Method: load checkpoint, build TWO models — one with bicubic skip (training
forward) and one with bilinear skip (ONNX-equivalent forward). Run both on
N held-out frames, compute mean PSNR vs ground truth.
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


class _ModelWithBilinearSkip(torch.nn.Module):
    """Wraps a trained SRCNNSimple, replacing the bicubic-antialias skip
    with bilinear-no-antialias to match the ONNX export path."""
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lr_rgb = x[:, :3, :, :]
        feat = F.relu(self.base.head_conv(x), inplace=True)
        feat = self.base.body(feat)
        residual = self.base.pixel_shuffle(self.base.upsample_conv(feat))
        bilinear = F.interpolate(
            lr_rgb, scale_factor=self.base.scale, mode="bilinear", align_corners=False
        )
        return bilinear + residual


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--scene", type=str, default="CitySample")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=16)
    args = p.parse_args()

    device = args.device
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    in_ch = int(ck.get("lean_in_channels", 12))

    bicubic_model = build_sr_model("simple", tier=tier, in_channels=in_ch, scale=2).to(device)
    bicubic_model.load_state_dict(ck["sr_model"])
    bicubic_model.train(False)
    bilinear_model = _ModelWithBilinearSkip(bicubic_model).to(device)
    bilinear_model.train(False)

    lr_synth = EngineAliasedLRSynth(
        enable_jitter=True, enable_taa_blur=True, enable_jpeg=True,
        jpeg_quality=85, blur_sigma=1.5,
    )
    candidates = [args.dataset_root, args.dataset_root / "srgd"]
    srgd_root = next((c for c in candidates if (c / "data" / "GameEngineData").is_dir()), None)
    if srgd_root is None:
        print(f"FAIL: no SRGD dataset under {candidates}")
        return 1

    ds = SRGDGaussianDataset(
        root=srgd_root, scale=2.0, lr_synth=lr_synth, scene=args.scene, force_synth_lr=True,
    )
    print(f"Held-out scene: {args.scene}  (n_frames={len(ds)})")
    print(f"Checkpoint: {args.checkpoint.name}  (tier={tier}, in_ch={in_ch})")

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2,
                        collate_fn=collate_examples, drop_last=True)

    bicubic_psnrs: list[float] = []
    bilinear_psnrs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            if len(bicubic_psnrs) >= args.n_samples:
                break
            lr = batch["lr_frame"].to(device)
            depth = batch["depth"].to(device)
            motion = batch["motion"].to(device)
            normals = batch["normals"].to(device)
            canvas = batch["canvas_hint"].to(device)
            gt_hr = batch["gt_hr_frame"].to(device)

            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            if in_ch == 9:
                x = x[:, :9]

            out_bicubic = bicubic_model(x).clamp(0, 1)
            out_bilinear = bilinear_model(x).clamp(0, 1)

            for b in range(lr.shape[0]):
                if len(bicubic_psnrs) >= args.n_samples:
                    break
                bicubic_psnrs.append(_psnr(out_bicubic[b], gt_hr[b]))
                bilinear_psnrs.append(_psnr(out_bilinear[b], gt_hr[b]))

    n = len(bicubic_psnrs)
    bicubic_mean = sum(bicubic_psnrs) / n
    bilinear_mean = sum(bilinear_psnrs) / n
    delta_mean = bilinear_mean - bicubic_mean
    deltas = [bi - bc for bi, bc in zip(bilinear_psnrs, bicubic_psnrs)]
    delta_max_loss = min(deltas)
    delta_max_gain = max(deltas)

    print()
    print(f"=== Bicubic-skip vs Bilinear-skip (ONNX-equivalent) on {args.scene} ===")
    print(f"n_samples           = {n}")
    print(f"bicubic mean PSNR   = {bicubic_mean:.3f} dB")
    print(f"bilinear mean PSNR  = {bilinear_mean:.3f} dB")
    print(f"delta (mean)        = {delta_mean:+.4f} dB")
    print(f"delta (worst loss)  = {delta_max_loss:+.4f} dB")
    print(f"delta (best gain)   = {delta_max_gain:+.4f} dB")
    print()
    if abs(delta_mean) < 0.05:
        print("VERDICT: negligible — ONNX bilinear path is fine for production.")
    elif delta_mean < -0.1:
        print("VERDICT: meaningful loss. Investigate alternative export (e.g., bake bicubic into graph).")
    else:
        print("VERDICT: small loss. Acceptable for v0; revisit if perceptual reviewers flag it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

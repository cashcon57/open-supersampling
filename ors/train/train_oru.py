"""ORU trainer (standalone, ``input_mode='rgb'``).

Targets are the GT renders; LR inputs are produced on-the-fly by bilinear
downsampling — keeps the training loop self-contained and matches T3's
floor-rounding upscale convention so HR shapes round-trip cleanly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ors.model import ORU
from .data import ORSDataset
from .losses import CompositeLoss


def _downsample(x: torch.Tensor, scale: float) -> torch.Tensor:
    """``int(H/scale)`` floor-division to match ORU's upscale rounding."""
    _, _, H, W = x.shape
    h = max(1, int(H / scale))
    w = max(1, int(W / scale))
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def _smoke_batches(n_steps: int, batch: int, hw: int, device: torch.device):
    for _ in range(n_steps):
        yield {
            "ground_truth": torch.randn(batch, 3, hw, hw, device=device).abs(),
            "depth":        torch.randn(batch, 1, hw, hw, device=device),
            "motion":       torch.randn(batch, 2, hw, hw, device=device),
        }


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train ORU (Open Ray Upscaler) — standalone.")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default="results/oru")
    p.add_argument("--tier", type=str, default="standard",
                   choices=["lite", "standard", "heavy"])
    p.add_argument("--scale", type=float, default=2.0,
                   choices=[1.3, 1.5, 1.7, 2.0])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ORU(input_mode="rgb", scale_factor=args.scale, tier=args.tier).to(device)
    loss_fn = CompositeLoss(w_lpips=0.0 if args.smoke_test else 0.05).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def step_once(target_hi, depth_hi, motion_hi):
        target_lo = _downsample(target_hi, args.scale)
        depth_lo  = _downsample(depth_hi,  args.scale)
        motion_lo = _downsample(motion_hi, args.scale)
        pred_hi = model(color=target_lo, depth=depth_lo, motion=motion_lo)
        # ORU floor-rounds the output; crop GT to the same spatial extent so
        # mismatched non-integer scales don't blow up the loss.
        _, _, H, W = pred_hi.shape
        return loss_fn(pred_hi, target_hi[:, :, :H, :W])

    if args.smoke_test:
        model.train()
        step = 0
        for batch in _smoke_batches(n_steps=200, batch=2, hw=64, device=device):
            loss = step_once(batch["ground_truth"], batch["depth"], batch["motion"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[oru:smoke] step {step:4d} loss={loss.item():.4f}")
            step += 1
        ckpt = out / "oru.pth"
        torch.save({"model": model.state_dict(), "tier": args.tier, "scale": args.scale}, ckpt)
        print(f"[oru:smoke] saved {ckpt}")
        return

    if args.data is None:
        raise SystemExit("--data is required unless --smoke-test")
    ds = ORSDataset(root=args.data, augment=True, crop_size=128)
    if len(ds) == 0:
        raise SystemExit(f"No samples found under {args.data}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = _move(batch, device)
            loss = step_once(batch["ground_truth"], batch["depth"], batch["motion"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[oru] epoch {epoch} step {step} loss={loss.item():.4f}")
            step += 1

    ckpt = out / "oru.pth"
    torch.save({"model": model.state_dict(), "tier": args.tier, "scale": args.scale}, ckpt)
    print(f"[oru] saved {ckpt}")


if __name__ == "__main__":
    main()

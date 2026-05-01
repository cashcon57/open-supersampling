"""Paired ORD+ORU trainer (two-stage freeze/unfreeze).

Stage A: ORD frozen (assumed pre-trained or warm-started via ``--ord-ckpt``);
optimize ORU only at full LR. ORU consumes ORD's 32-ch FP16 features via the
handoff contract.

Stage B: Unfreeze ORD; optimize the whole pipeline at ``lr * 0.2``. Loss is
``loss(rgb_lo, target_lo) + loss(rgb_hi, target_hi)`` with target_lo built by
the same floor-rounded downsample used in :mod:`oss.train.train_sr`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oss.model import ORD, ORU
from oss.model.adapter import PairedORS
from .data import ORSDataset
from .losses import CompositeLoss


def _downsample(x: torch.Tensor, scale: float) -> torch.Tensor:
    _, _, H, W = x.shape
    h = max(1, int(H / scale))
    w = max(1, int(W / scale))
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def _smoke_batches(n_steps: int, batch: int, hw_lo: int, scale: float, device: torch.device):
    """Smoke batches feed LR-shape inputs to ORD and HR-shape targets."""
    hw_hi = int(hw_lo * scale)
    for _ in range(n_steps):
        yield {
            "noisy":   torch.randn(batch, 3,  hw_lo, hw_lo, device=device).abs(),
            "aux":     torch.randn(batch, 11, hw_lo, hw_lo, device=device),
            "history": torch.randn(batch, 3,  hw_lo, hw_lo, device=device).abs(),
            "depth":   torch.randn(batch, 1,  hw_lo, hw_lo, device=device),
            "motion":  torch.randn(batch, 2,  hw_lo, hw_lo, device=device),
            "ground_truth_hi": torch.randn(batch, 3, hw_hi, hw_hi, device=device).abs(),
        }


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _step(pair: PairedORS, batch: dict, scale: float, loss_fn: CompositeLoss) -> torch.Tensor:
    target_hi = batch["ground_truth_hi"]
    target_lo = _downsample(target_hi, scale)
    rgb_lo, rgb_hi = pair(
        noisy=batch["noisy"], aux=batch["aux"], history=batch["history"],
        depth=batch["depth"], motion=batch["motion"],
    )
    # Crop both to the predicted shape (ORU floor-rounds, target_lo is already LR).
    _, _, Hl, Wl = rgb_lo.shape
    _, _, Hh, Wh = rgb_hi.shape
    return (
        loss_fn(rgb_lo, target_lo[:, :, :Hl, :Wl])
        + loss_fn(rgb_hi, target_hi[:, :, :Hh, :Wh])
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train Paired ORD+ORU (two-stage).")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default="results/paired")
    p.add_argument("--tier", type=str, default="standard",
                   choices=["lite", "standard", "heavy"])
    p.add_argument("--scale", type=float, default=2.0,
                   choices=[1.3, 1.5, 1.7, 2.0])
    p.add_argument("--ord-ckpt", type=str, default=None,
                   help="Optional ORD warm-start checkpoint (from train_ord).")
    p.add_argument("--epochs-stage-a", type=int, default=5)
    p.add_argument("--epochs-stage-b", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ord_model = ORD(tier=args.tier).to(device)
    oru_model = ORU(input_mode="features", scale_factor=args.scale, tier=args.tier).to(device)

    if args.ord_ckpt:
        sd = torch.load(args.ord_ckpt, map_location=device)
        ord_model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"[paired] warm-started ORD from {args.ord_ckpt}")

    pair = PairedORS(ord_model, oru_model).to(device)
    loss_fn = CompositeLoss(w_lpips=0.0 if args.smoke_test else 0.05).to(device)

    def stage_a():
        # Freeze ORD; optimize ORU only.
        for p_ in ord_model.parameters():
            p_.requires_grad = False
        opt = torch.optim.Adam(oru_model.parameters(), lr=args.lr)

        if args.smoke_test:
            pair.train()
            for step, batch in enumerate(_smoke_batches(100, 2, 32, args.scale, device)):
                loss = _step(pair, batch, args.scale, loss_fn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    print(f"[paired:A:smoke] step {step:4d} loss={loss.item():.4f}")
            return

        ds = ORSDataset(root=args.data, augment=True, crop_size=128)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        pair.train()
        step = 0
        for epoch in range(args.epochs_stage_a):
            for batch in dl:
                batch = _move(batch, device)
                # Real dataset: noisy/aux/history/depth/motion are LR; we
                # synthesize HR target by upsampling GT to (H*scale, W*scale).
                target_hi = F.interpolate(
                    batch["ground_truth"],
                    scale_factor=args.scale,
                    mode="bilinear",
                    align_corners=False,
                )
                batch["ground_truth_hi"] = target_hi
                loss = _step(pair, batch, args.scale, loss_fn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    print(f"[paired:A] epoch {epoch} step {step} loss={loss.item():.4f}")
                step += 1

    def stage_b():
        # Unfreeze; whole-pipeline at reduced LR.
        for p_ in ord_model.parameters():
            p_.requires_grad = True
        opt = torch.optim.Adam(pair.parameters(), lr=args.lr * 0.2)

        if args.smoke_test:
            pair.train()
            for step, batch in enumerate(_smoke_batches(100, 2, 32, args.scale, device)):
                loss = _step(pair, batch, args.scale, loss_fn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    print(f"[paired:B:smoke] step {step:4d} loss={loss.item():.4f}")
            return

        ds = ORSDataset(root=args.data, augment=True, crop_size=128)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        pair.train()
        step = 0
        for epoch in range(args.epochs_stage_b):
            for batch in dl:
                batch = _move(batch, device)
                target_hi = F.interpolate(
                    batch["ground_truth"],
                    scale_factor=args.scale,
                    mode="bilinear",
                    align_corners=False,
                )
                batch["ground_truth_hi"] = target_hi
                loss = _step(pair, batch, args.scale, loss_fn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    print(f"[paired:B] epoch {epoch} step {step} loss={loss.item():.4f}")
                step += 1

    if not args.smoke_test and args.data is None:
        raise SystemExit("--data is required unless --smoke-test")

    print("[paired] === Stage A: ORD frozen ===")
    stage_a()
    print("[paired] === Stage B: full unfreeze ===")
    stage_b()

    ckpt = out / "paired.pth"
    torch.save({
        "ord": ord_model.state_dict(),
        "oru": oru_model.state_dict(),
        "tier": args.tier,
        "scale": args.scale,
    }, ckpt)
    print(f"[paired] saved {ckpt}")


if __name__ == "__main__":
    main()

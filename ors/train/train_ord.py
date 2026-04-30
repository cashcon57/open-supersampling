"""ORD trainer.

Smoke mode: 200 synthetic random batches at 64x64 batch-2, no on-disk data,
writes ``<out>/ord.pth`` with ``{model: state_dict, tier}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ors.model import ORD
from .data import ORSDataset
from .losses import CompositeLoss


def _smoke_batches(n_steps: int, batch: int, hw: int, device: torch.device):
    for _ in range(n_steps):
        yield {
            "noisy":   torch.randn(batch, 3,  hw, hw, device=device).abs(),
            "ground_truth": torch.randn(batch, 3,  hw, hw, device=device).abs(),
            "aux":     torch.randn(batch, 11, hw, hw, device=device),
            "history": torch.randn(batch, 3,  hw, hw, device=device).abs(),
        }


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train ORD (Open Ray Denoiser).")
    p.add_argument("--data", type=str, default=None, help="EXR dataset root")
    p.add_argument("--out", type=str, default="results/ord")
    p.add_argument("--tier", type=str, default="standard",
                   choices=["lite", "standard", "heavy"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 200 random-tensor steps and exit (CPU-friendly).")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ORD(tier=args.tier).to(device)
    # LPIPS pulls a VGG download — skip in smoke tests.
    loss_fn = CompositeLoss(w_lpips=0.0 if args.smoke_test else 0.05).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.smoke_test:
        model.train()
        step = 0
        for batch in _smoke_batches(n_steps=200, batch=2, hw=64, device=device):
            rgb, _ = model(batch["noisy"], batch["aux"], batch["history"])
            loss = loss_fn(rgb, batch["ground_truth"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[ord:smoke] step {step:4d} loss={loss.item():.4f}")
            step += 1
        ckpt = out / "ord.pth"
        torch.save({"model": model.state_dict(), "tier": args.tier}, ckpt)
        print(f"[ord:smoke] saved {ckpt}")
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
            rgb, _ = model(batch["noisy"], batch["aux"], batch["history"])
            loss = loss_fn(rgb, batch["ground_truth"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[ord] epoch {epoch} step {step} loss={loss.item():.4f}")
            step += 1

    ckpt = out / "ord.pth"
    torch.save({"model": model.state_dict(), "tier": args.tier}, ckpt)
    print(f"[ord] saved {ckpt}")


if __name__ == "__main__":
    main()

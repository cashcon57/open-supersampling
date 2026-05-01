"""OSS-RG trainer.

Smoke mode: 200 synthetic random batches at 64x64, no on-disk data.
Data modes:
  --data <exr-root>       EXR triplets from mitsuba_pipeline (existing)
  --noisebase <nb-root>   NoiseBase zarr ZipStore sequences
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from oss.model import OSSRG
from .data import ORSDataset
from .losses import CompositeLoss


def _smoke_batches(n_steps: int, batch: int, hw: int, device: torch.device):
    for _ in range(n_steps):
        yield {
            "noisy":        torch.randn(batch, 3,  hw, hw, device=device).abs(),
            "ground_truth": torch.randn(batch, 3,  hw, hw, device=device).abs(),
            "aux":          torch.randn(batch, 11, hw, hw, device=device),
            "history":      torch.randn(batch, 3,  hw, hw, device=device).abs(),
        }


def _nb_to_rg_batch(batch: dict) -> dict:
    return {
        "noisy":        batch["noisy"],
        "ground_truth": batch["gt"],
        "aux":          batch["aux"],
        "history":      batch["history"],
    }


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train OSS-RG (ray denoiser).")
    p.add_argument("--data", type=str, default=None,
                   help="EXR dataset root (Mitsuba pipeline output)")
    p.add_argument("--noisebase", type=str, default=None,
                   help="NoiseBase root directory (zarr ZipStore sequences)")
    p.add_argument("--out", type=str, default="results/oss_rg")
    p.add_argument("--tier", type=str, default="standard",
                   choices=["lite", "standard", "heavy"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 200 random-tensor steps and exit (CPU-friendly).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OSSRG(tier=args.tier).to(device)
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
                print(f"[oss_rg:smoke] step {step:4d} loss={loss.item():.4f}")
            step += 1
        ckpt = out / "oss_rg.pth"
        torch.save({"model": model.state_dict(), "tier": args.tier}, ckpt)
        print(f"[oss_rg:smoke] saved {ckpt}")
        return

    if args.data is None and args.noisebase is None:
        raise SystemExit("Provide --data (EXR) or --noisebase (zarr) or --smoke-test")

    datasets = []
    if args.data:
        ds_exr = ORSDataset(root=args.data, augment=True, crop_size=128)
        if len(ds_exr) == 0:
            raise SystemExit(f"No EXR samples found under {args.data}")
        datasets.append(ds_exr)

    if args.noisebase:
        from oss.data.noisebase_rg import NoiseBaseRGDataset
        ds_nb = NoiseBaseRGDataset(root=args.noisebase, split="train")
        datasets.append(ds_nb)
        print(f"[oss_rg] NoiseBase: {len(ds_nb)} frames")

    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for raw_batch in dl:
            batch = _nb_to_rg_batch(raw_batch) if args.noisebase and not args.data else raw_batch
            batch = _move(batch, device)
            rgb, _ = model(batch["noisy"], batch["aux"], batch["history"])
            loss = loss_fn(rgb, batch["ground_truth"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[oss_rg] epoch {epoch} step {step} loss={loss.item():.4f}")
            step += 1

    ckpt = out / "oss_rg.pth"
    torch.save({"model": model.state_dict(), "tier": args.tier}, ckpt)
    print(f"[oss_rg] saved {ckpt}")


if __name__ == "__main__":
    main()

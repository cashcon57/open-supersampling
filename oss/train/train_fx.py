"""OSS-FX trainer: α-conditioned frame extrapolation.

Loads SintelFxDataset + Vimeo90kFxDataset, concatenates, and trains OSSFx
with a curriculum that starts at α ∈ [0.4, 0.6] for the first 10% of
epochs, then expands to the full range.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from oss.model.oss_fx import HISTORY_CH, OSSFx
from oss.train.losses_fx import extrapolation_loss


def _synthetic_loader(steps: int, batch_size: int) -> Iterable[dict]:
    H, W = 32, 32
    for _ in range(steps):
        B = batch_size
        alpha = torch.rand(B) * 0.85 + 0.1
        yield {
            "warped":  torch.randn(B, 3, H, W).abs().clamp(0.0, 1.0),
            "depth":   torch.rand(B, 1, H, W),
            "history": torch.zeros(B, HISTORY_CH, H, W),
            "alpha":   alpha,
            "target":  torch.randn(B, 3, H, W).abs().clamp(0.0, 1.0),
            "frame_t": torch.randn(B, 3, H, W).abs().clamp(0.0, 1.0),
        }


def _alpha_curriculum(epoch: int, total_epochs: int) -> tuple[float, float]:
    warmup = max(1, int(total_epochs * 0.1))
    if epoch < warmup:
        return 0.4, 0.6
    return 0.1, 0.95


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def train_fx(
    sintel_root: str,
    vimeo_root: str,
    out_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cuda",
    smoke: bool = False,
) -> None:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = OSSFx().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if smoke:
        loader = list(_synthetic_loader(steps=10, batch_size=batch_size))
        epochs = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
    else:
        from oss.data.sintel_fx import SintelFxDataset
        from oss.data.vimeo90k_fx import Vimeo90kFxDataset

        sintel_ds = SintelFxDataset(root=sintel_root, split="train", augment=True)
        vimeo_ds = Vimeo90kFxDataset(root=vimeo_root, split="train", augment=True)
        combined = ConcatDataset([sintel_ds, vimeo_ds])
        loader = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=_collate,
            drop_last=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    log_interval = 5 if smoke else 50
    model.train()

    for epoch in range(epochs):
        alpha_lo, alpha_hi = _alpha_curriculum(epoch, epochs)

        for step, batch in enumerate(tqdm(loader, desc=f"fx ep{epoch}")):
            warped  = batch["warped"].to(dev)
            depth   = batch["depth"].to(dev)
            history = batch["history"].to(dev)
            target  = batch["target"].to(dev)

            if smoke:
                alpha = batch["alpha"].to(dev)
            else:
                alpha = torch.empty(warped.shape[0], device=dev).uniform_(alpha_lo, alpha_hi)

            pred, new_history = model(warped, depth, history, alpha)
            pred_prev = warped

            loss = extrapolation_loss(pred, target, pred_prev, alpha)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if step % log_interval == 0:
                last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
                print(f"ep{epoch} step{step} loss={loss.item():.4f} alpha=[{alpha_lo:.2f},{alpha_hi:.2f}] lr={last_lr:.2e}")

        scheduler.step()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "config": {"history_ch": HISTORY_CH},
            },
            out_path / "oss_fx.pth",
        )

    print(f"saved {out_path / 'oss_fx.pth'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train OSS-FX (α-conditioned frame extrapolation).")
    p.add_argument("--sintel", type=str, default="data/sintel")
    p.add_argument("--vimeo",  type=str, default="data/vimeo90k")
    p.add_argument("--out",    type=str, default="results/fx_run1")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke",  action="store_true")
    args = p.parse_args()
    train_fx(
        sintel_root=args.sintel,
        vimeo_root=args.vimeo,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()

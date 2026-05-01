"""ORU-Pico trainer (temporal sequences + composite loss).

Pico is the v0.2 Steam-Deck-tier model that consumes temporal sequences from
NoiseBase. Training unrolls T timesteps with hidden-state propagation across
frames; loss is per-frame ``relative_l2 + 0.1*(1-SSIM)`` plus an inter-frame
``0.1 * temporal_consistency_loss(pred_t, pred_{t-1}, motion_lr_t)``.

LPIPS is intentionally disabled here — VGG download cost is heavy for smoke
runs and v0.2-pico's HR target (480p/720p/1080p) is small enough that the
relative-L2 + SSIM combination is adequate for the initial trainer. Bring
LPIPS back in v0.2-beta once we have a real validation loop.

Hidden state is propagated forward in the time loop; predictions are
``detach()``-ed before being threaded back as next frame's history to bound
BPTT depth at one timestep (matches MUNet 2025 / NPPD recurrent-cell SOP).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from oss.data.noisebase import NoiseBaseDataset
from oss.model.oru_pico import OSSPico
from oss.train.losses import CompositeLoss, temporal_consistency_loss


def _synthetic_sequence_loader(steps: int, sequence_length: int = 8) -> Iterable[dict]:
    """Synthesize random sequences for smoke-test mode.

    Yields ``steps`` batches of shape ``(B=1, T, C, H, W)`` matching the
    NoiseBaseDataset adapter contract. Spatial size is the smallest the model
    supports while keeping all four U-Net levels intact (32x32 LR -> 64x64 HR).
    """
    H_lr, W_lr = 32, 32
    H_hr, W_hr = H_lr * 2, W_lr * 2
    for _ in range(steps):
        yield {
            "color_lr": torch.randn(1, sequence_length, 3, H_lr, W_lr).abs(),
            "gt_hr": torch.randn(1, sequence_length, 3, H_hr, W_hr).abs(),
            "motion_lr": torch.randn(1, sequence_length, 2, H_lr, W_lr) * 0.1,
            "depth_lr": torch.randn(1, sequence_length, 1, H_lr, W_lr),
            "normals_lr": torch.randn(1, sequence_length, 3, H_lr, W_lr),
            "albedo_lr": torch.randn(1, sequence_length, 3, H_lr, W_lr),
        }


def _unroll_sequence(
    model: OSSPico,
    batch: dict,
    scale_factor: float,
    loss_fn: CompositeLoss,
    device: torch.device,
) -> torch.Tensor:
    """Walk T timesteps, accumulating per-frame loss + temporal consistency.

    History is initialized to zeros at t=0; predictions are detached before
    being passed forward to bound BPTT to a single step. Mean-over-T is
    returned so the loss magnitude is independent of sequence length.
    """
    T = batch["color_lr"].shape[1]
    gt0 = batch["gt_hr"][:, 0]
    history_hr = torch.zeros_like(gt0).to(device)
    hidden = None
    total_loss = torch.tensor(0.0, device=device)
    prev_pred: torch.Tensor | None = None

    for t in range(T):
        color_lr = batch["color_lr"][:, t].to(device)
        depth_lr = batch["depth_lr"][:, t].to(device)
        motion_lr = batch["motion_lr"][:, t].to(device)
        normals_lr = batch["normals_lr"][:, t].to(device)
        albedo_lr = batch["albedo_lr"][:, t].to(device)
        gt_hr = batch["gt_hr"][:, t].to(device)

        rgb_pred, hidden = model(
            color_lr, depth_lr, motion_lr, normals_lr, albedo_lr, history_hr, hidden
        )
        per_frame_loss = loss_fn(rgb_pred, gt_hr)
        if prev_pred is not None:
            per_frame_loss = per_frame_loss + 0.1 * temporal_consistency_loss(
                rgb_pred, prev_pred, motion_lr, scale_factor=scale_factor
            )
        total_loss = total_loss + per_frame_loss

        # Detach both history and prev_pred to cap BPTT at one timestep.
        history_hr = rgb_pred.detach()
        prev_pred = rgb_pred.detach()

    return total_loss / T


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OSSPico().to(device)
    # LPIPS disabled in pico trainer; see module docstring.
    loss_fn = CompositeLoss(w_l2=1.0, w_ssim=0.1, w_lpips=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        loader = list(
            _synthetic_sequence_loader(steps=50, sequence_length=args.sequence_length)
        )
        epochs = 1
    else:
        ds = NoiseBaseDataset(
            root=Path(args.data),
            sequence_length=args.sequence_length,
            scale_factor=args.scale_factor,
            split="train",
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        epochs = args.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    log_interval = 10 if args.smoke_test else 50

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(loader, desc=f"pico ep{epoch}")):
            loss = _unroll_sequence(model, batch, args.scale_factor, loss_fn, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % log_interval == 0:
                print(f"ep{epoch} step{step} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "config": {
                    "scale_factor": args.scale_factor,
                    "tier": "pico",
                },
            },
            out_dir / "oru_pico.pth",
        )
    print(f"saved {out_dir / 'oru_pico.pth'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train ORU-Pico (temporal recurrent).")
    p.add_argument("--data", type=str, default="data/noisebase")
    p.add_argument("--out", type=str, default="results/pico")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--scale-factor", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

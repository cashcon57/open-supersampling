"""v7 training scaffold (Phase 2A).

Wires V7Model + TartanAirIntermediateTriplets + oss_fx_loss into a
runnable training loop. Saves checkpoints + metrics; emits per-step
loss components for dashboard ingestion.

IMPORTANT PHASE 2A CAVEAT:
  This scaffold does NOT yet spawn Gaussians from backbone features --
  the canvas stays empty throughout training, so the model effectively
  trains as a 2D SR model. The OSS-FX loss components ARE computed
  (intermediate-frame prediction) but at canvas=empty, the
  intermediate prediction is just the SR output at the same LR input
  -- gives gradient signal to the backbone but cannot exercise the
  N-D time-slice math.

  Phase 2B adds the canvas spawner: a learnable module that decodes
  backbone features into Gaussian params and adds them to the canvas
  at t = current frame, allowing the time-slice to produce meaningful
  intermediate frames.

  Phase 2A validates: imports work, data pipeline works, loss runs,
  gradients flow, checkpoints save. End-to-end smoke test of the
  v7 training stack.

Usage (will need a GPU host with TartanAir extracted):
  python scripts/sr_train_v7.py \\
      --tartanair-root E:/datasets/tartanair_extracted \\
      --output-dir E:/checkpoints/srcnn-v7.0-smoke \\
      --steps 100 --batch-size 2 --device cuda --log-every 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.intermediate_dataset import TartanAirIntermediateTriplets
from oss.sr.v7.losses import oss_fx_loss


def _device(arg: str) -> str:
    if arg == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        return "cpu"
    return arg


def collate_triplets(batch: list[dict]) -> dict:
    """Stack a list of triplets into batched tensors. Each triplet is
    a dict with frame-N / frame-N-half / frame-N+1 sub-dicts."""
    n_lr = torch.stack([b["n"]["lr"] for b in batch])
    n_depth = torch.stack([b["n"]["depth"] for b in batch])
    n_motion = torch.stack([b["n"]["motion"] for b in batch])
    n_normals = torch.stack([b["n"]["normals"] for b in batch])
    n_gt = torch.stack([b["n"]["gt_hr"] for b in batch])
    n_half_gt = torch.stack([b["n_half"]["gt_hr"] for b in batch])
    np1_lr = torch.stack([b["n_plus_1"]["lr"] for b in batch])
    np1_depth = torch.stack([b["n_plus_1"]["depth"] for b in batch])
    np1_motion = torch.stack([b["n_plus_1"]["motion"] for b in batch])
    np1_normals = torch.stack([b["n_plus_1"]["normals"] for b in batch])
    np1_gt = torch.stack([b["n_plus_1"]["gt_hr"] for b in batch])
    motion_n_to_np1 = torch.stack([b["motion_n_to_np1"] for b in batch])
    return {
        "n_lr": n_lr,
        "n_depth": n_depth,
        "n_motion": n_motion,
        "n_normals": n_normals,
        "n_gt": n_gt,
        "n_half_gt": n_half_gt,
        "np1_lr": np1_lr,
        "np1_depth": np1_depth,
        "np1_motion": np1_motion,
        "np1_normals": np1_normals,
        "np1_gt": np1_gt,
        "motion_n_to_np1": motion_n_to_np1,
    }


def build_9ch_input(lr: torch.Tensor, depth: torch.Tensor, motion: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Stack the 9-channel input the v7 backbone expects."""
    return torch.cat([lr, depth, motion, normals], dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tartanair-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feat-dim", type=int, default=32)
    parser.add_argument("--latent-rank", type=int, default=16)
    parser.add_argument("--canvas-capacity", type=int, default=4096)
    parser.add_argument("--backbone-blocks", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--lambda-charbonnier", type=float, default=1.0)
    parser.add_argument("--lambda-lpips", type=float, default=1.0)
    parser.add_argument("--lambda-fg", type=float, default=1.0)
    parser.add_argument("--lambda-fg-lpips", type=float, default=0.5)
    parser.add_argument("--lambda-temp-consistency", type=float, default=0.1)
    parser.add_argument("--max-triplets", type=int, default=None,
                        help="Cap dataset size (useful for smoke testing).")
    args = parser.parse_args()

    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "history.jsonl").touch()

    # Build model
    cfg = V7Config(
        in_channels=9,
        scale=2,
        feat_dim=args.feat_dim,
        latent_rank=args.latent_rank,
        canvas_capacity=args.canvas_capacity,
        backbone_blocks=args.backbone_blocks,
    )
    model = V7Model(cfg).to(device)
    model.allocate_canvas(device)
    model.train(True)

    # Build dataset
    from oss.gaussian.data import TartanAirGaussianDataset
    base = TartanAirGaussianDataset(root=args.tartanair_root, scale=2.0)
    ds = TartanAirIntermediateTriplets(base, max_triplets=args.max_triplets)
    print(f"[train] dataset: {len(ds)} triplets")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_triplets, drop_last=True,
    )

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    t0 = time.perf_counter()
    history_path = args.output_dir / "history.jsonl"
    step = 0
    while step < args.steps:
        for batch in loader:
            step += 1
            if step > args.steps:
                break
            optim.zero_grad()

            # Phase 2B: BackboneSpawner is B=1 only (per-rank canvas
            # state). We accumulate loss across the batch by looping
            # over samples and dividing.
            batch_total_loss = None
            batch_parts: dict[str, float] = {}
            n_samples = batch["n_lr"].shape[0]

            for b in range(n_samples):
                # Per-sample reset of canvas (each trajectory pair is
                # an independent canvas trajectory for now).
                model.reset_state(device)

                n_lr_in_b = build_9ch_input(
                    batch["n_lr"][b:b+1].to(device),
                    batch["n_depth"][b:b+1].to(device),
                    batch["n_motion"][b:b+1].to(device),
                    batch["n_normals"][b:b+1].to(device),
                )
                np1_lr_in_b = build_9ch_input(
                    batch["np1_lr"][b:b+1].to(device),
                    batch["np1_depth"][b:b+1].to(device),
                    batch["np1_motion"][b:b+1].to(device),
                    batch["np1_normals"][b:b+1].to(device),
                )
                n_gt_b = batch["n_gt"][b:b+1].to(device).clamp(0, 1)
                n_half_gt_b = batch["n_half_gt"][b:b+1].to(device).clamp(0, 1)
                np1_gt_b = batch["np1_gt"][b:b+1].to(device).clamp(0, 1)

                # Forward at frame N: SPAWN Gaussians into canvas at t=0
                out_main_n = model(n_lr_in_b, t_query=0.0, spawn_at_t=0.0)
                # Forward at frame N+1: spawn at t=2 so the (i, i+2)
                # spacing matches the dataset's triplet convention
                # (alpha=0.5 lives at t=1, between the two spawned
                # times).
                out_main_np1 = model(np1_lr_in_b, t_query=2.0, spawn_at_t=2.0)
                # Render at intermediate t=1 (alpha=0.5 between t=0 and t=2);
                # no new spawn, uses canvas content from previous two
                # forward passes.
                out_inter = model(np1_lr_in_b, t_query=1.0)

                loss_b, parts_b = oss_fx_loss(
                    out_main=out_main_np1,
                    gt_main=np1_gt_b,
                    out_inter_list=[out_inter],
                    gt_inter_list=[n_half_gt_b],
                    lambda_charbonnier=args.lambda_charbonnier,
                    lambda_lpips=args.lambda_lpips,
                    lambda_fg=args.lambda_fg,
                    lambda_fg_lpips=args.lambda_fg_lpips,
                    lambda_temp_consistency=0.0,
                )
                if batch_total_loss is None:
                    batch_total_loss = loss_b
                else:
                    batch_total_loss = batch_total_loss + loss_b
                for k, v in parts_b.items():
                    batch_parts[k] = batch_parts.get(k, 0.0) + float(v)

            batch_total_loss = batch_total_loss / float(n_samples)
            for k in batch_parts:
                batch_parts[k] /= float(n_samples)
            batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            loss = batch_total_loss
            parts = batch_parts

            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                parts["step"] = step
                parts["elapsed_s"] = elapsed
                print(
                    f"[step {step:5d}] loss={parts['total']:.4f} "
                    f"sr_char={parts.get('sr_charbonnier', 0.0):.4f} "
                    f"fg_char={parts.get('fg_charbonnier', 0.0):.4f} "
                    f"elapsed={elapsed:.0f}s"
                )
                with open(history_path, "a") as f:
                    f.write(json.dumps(parts) + "\n")

            if step % args.ckpt_every == 0:
                ckpt_path = args.output_dir / f"step-{step:08d}.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "cfg": vars(cfg),
                    "args": vars(args),
                }, ckpt_path)
                print(f"[train] ckpt -> {ckpt_path}")

    # Final ckpt
    final_path = args.output_dir / f"step-{step:08d}-final.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "cfg": vars(cfg),
        "args": vars(args),
    }, final_path)
    print(f"[train] done -> {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

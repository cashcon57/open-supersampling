#!/usr/bin/env python
"""v6 training script. Consumes ``oss.sr.v6`` modules.

Mirrors the v5 trainers' structure (auto-resume, metrics dump, dashboard
compat) but with v6's loss recipe, EMA, cosine LR + warm restarts, GAN
warmup, and mixed TartanAir + Hypersim dataset.

Smoke mode (``--smoke``) runs five CPU-friendly synthetic steps by default.
It still exercises the same trainer wiring: patch sampling, G/D optimizers,
EMA, scheduler stepping, metrics JSONL, checkpoints, and auto-resume.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional

# Allow ``python scripts/sr_train_v6.py`` to import ``oss.*`` without
# ``pip install -e .`` first. Mirrors sr_train_gaussian_temporal.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

from oss.sr.v6.dataset import build_v6_training_dataset  # noqa: E402
from oss.sr.v6.discriminator import UNetDiscriminator  # noqa: E402
from oss.sr.v6.ema import EMAModel  # noqa: E402
from oss.sr.v6.losses import V6CompositeLoss, gan_hinge_d_loss  # noqa: E402
from oss.sr.v6.patch_sampling import importance_weighted_patch_indices  # noqa: E402
from oss.sr.v6.schedules import CosineLRWithWarmRestarts  # noqa: E402

log = logging.getLogger("oss.sr.v6.train")


# ---------------------------------------------------------------------------
# DDP helpers (mirror v5)
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def _ddp_init_if_needed(device_arg: str) -> tuple[str, int, int]:
    if not _is_distributed():
        return device_arg, 0, 1

    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return device, rank, world_size


def _ddp_cleanup() -> None:
    if _is_distributed():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for checkpoints + metrics.json.")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="Root of the extracted TartanAir dataset.")
    p.add_argument("--hypersim-root", type=Path, default=None,
                   help="Root of the extracted Hypersim dataset (apple/ml-hypersim).")
    p.add_argument("--held-out-envs", type=str, nargs="*", default=("oldtown",),
                   help="TartanAir env names to exclude from training.")
    p.add_argument("--held-out-scenes", type=str, nargs="*", default=None,
                   help="Hypersim scene names to exclude from training.")

    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=20_000,
                   help="GAN warmup until this step (pixel-only before).")
    p.add_argument("--T0", type=int, default=50_000,
                   help="Cosine warm-restart period (T_mult=1).")
    p.add_argument(
        "--num-restarts", type=int, default=5,
        help="Cosine warm restarts. Default 5 makes the schedule cover "
             "6 cycles x T_0 = 300K steps with T_0=50000 (memo §6 max_steps). "
             "Audit found default=3 only covered 200K, leaving 100K at LR=0.",
    )

    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None,
                   help="HR patch size; LR patches are patch_size / scale.")
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--trajectory-length", type=int, default=None,
                   help="Number of consecutive frames per training trajectory.")

    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--backbone", choices=("hat-tiny", "hat-small", "hat-l"),
                   default=None)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="HAT-L SA1B warm-start ckpt (from GSASR) — optional.")

    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--first-ckpt-step", type=int, default=100,
                   help="Always write an early checkpoint at this global step.")
    p.add_argument("--log-every", type=int, default=20)

    p.add_argument("--hypersim-mix-ratio", type=float, default=0.333)
    p.add_argument("--tartanair-mix-ratio", type=float, default=0.667)

    p.add_argument("--smoke", action="store_true",
                   help="Synthetic random tensors, no datasets. CI / pre-launch sanity.")
    return p.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_steps is None:
        args.max_steps = 5 if args.smoke else 300_000
    if args.backbone is None:
        args.backbone = "hat-tiny" if args.smoke else "hat-l"
    if args.patch_size is None:
        args.patch_size = 32 if args.smoke else 256
    if args.batch_size is None:
        args.batch_size = 1 if args.smoke else 4
    if args.grad_accum is None:
        args.grad_accum = 1 if args.smoke else 4
    if args.trajectory_length is None:
        args.trajectory_length = 2 if args.smoke else 4
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.trajectory_length < 1:
        raise ValueError("--trajectory-length must be >= 1")
    if args.patch_size < 32:
        raise ValueError("--patch-size must be >= 32 so VGG relu5_1 remains valid")
    return args


# ---------------------------------------------------------------------------
# Synthetic smoke dataset
# ---------------------------------------------------------------------------


class SyntheticV6Dataset(Dataset):
    """Small deterministic v6 trajectory dataset for smoke tests."""

    def __init__(
        self,
        length: int,
        hr_size: int,
        scale: int = 2,
        seed: int = 0,
        trajectory_length: int = 2,
    ) -> None:
        self.length = int(length)
        self.hr_size = int(hr_size)
        self.scale = int(scale)
        self.seed = int(seed)
        self.trajectory_length = int(trajectory_length)

    def __len__(self) -> int:
        return self.length

    def _frame(self, traj_idx: int, frame_idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(
            self.seed + int(traj_idx) * 1009 + int(frame_idx)
        )
        h_hr = w_hr = self.hr_size
        h_lr = h_hr // self.scale
        w_lr = w_hr // self.scale
        gt = torch.rand(3, h_hr, w_hr, generator=g)
        if frame_idx > 0:
            # Keep adjacent frames related so temporal consistency is finite
            # and non-zero in smoke without needing a real optical-flow field.
            base_g = torch.Generator().manual_seed(self.seed + int(traj_idx) * 1009)
            base_gt = torch.rand(3, h_hr, w_hr, generator=base_g)
            gt = (0.85 * base_gt + 0.15 * gt).clamp(0.0, 1.0)
        lr = F.interpolate(
            gt.unsqueeze(0), size=(h_lr, w_lr), mode="bilinear", align_corners=False,
        ).squeeze(0)
        depth = torch.rand(1, h_lr, w_lr, generator=g)
        motion = torch.zeros(2, h_lr, w_lr)
        normals = torch.rand(3, h_lr, w_lr, generator=g) * 2.0 - 1.0
        canvas = torch.zeros(3, h_lr, w_lr)
        return {
            "lr_frame": lr.float(),
            "gt_hr_frame": gt.float(),
            "depth": depth.float(),
            "motion": motion.float(),
            "normals": normals.float(),
            "canvas_hint": canvas.float(),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frames = [self._frame(int(idx), t) for t in range(self.trajectory_length)]
        return {
            "lr_frame": torch.stack([f["lr_frame"] for f in frames], dim=0),
            "gt_hr_frame": torch.stack([f["gt_hr_frame"] for f in frames], dim=0),
            "depth": torch.stack([f["depth"] for f in frames], dim=0),
            "motion": torch.stack([f["motion"] for f in frames], dim=0),
            "normals": torch.stack([f["normals"] for f in frames], dim=0),
            "canvas_hint": torch.stack([f["canvas_hint"] for f in frames], dim=0),
            "metadata": {
                "source": "synthetic",
                "idx": int(idx),
                "trajectory_length": self.trajectory_length,
            },
        }


# ---------------------------------------------------------------------------
# Batch preparation + patch sampling
# ---------------------------------------------------------------------------


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("lr_frame", "gt_hr_frame", "depth", "motion", "normals", "canvas_hint"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def _crop_chw(x: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    return x[..., top:top + size, left:left + size]


def sample_v6_patch_batch(
    batch: dict[str, torch.Tensor],
    *,
    hr_patch_size: int,
    scale: int,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Importance-sample one aligned LR/HR patch per item in a batch."""
    lr = batch["lr_frame"]
    if lr.dim() == 5:
        return sample_v6_trajectory_patch_batch(
            batch,
            hr_patch_size=hr_patch_size,
            scale=scale,
            generator=generator,
        )
    target = batch["gt_hr_frame"]
    depth = batch["depth"]
    motion = batch["motion"]
    normals = batch["normals"]
    canvas = batch["canvas_hint"]

    if target.shape[-1] < hr_patch_size or target.shape[-2] < hr_patch_size:
        raise ValueError(
            f"target spatial size {tuple(target.shape[-2:])} smaller than "
            f"patch_size={hr_patch_size}"
        )
    if hr_patch_size % scale != 0:
        raise ValueError(f"patch_size={hr_patch_size} must be divisible by scale={scale}")
    lr_patch_size = hr_patch_size // scale

    lr_patches = []
    target_patches = []
    depth_patches = []
    motion_patches = []
    normal_patches = []
    canvas_patches = []
    coords: list[tuple[int, int]] = []

    for i in range(lr.shape[0]):
        top_lr, left_lr = importance_weighted_patch_indices(
            lr[i].detach().float().cpu(),
            patch_size=lr_patch_size,
            num_patches=1,
            importance_ratio=0.7,
            generator=generator,
        )[0]
        top_hr = top_lr * scale
        left_hr = left_lr * scale
        coords.append((top_hr, left_hr))
        lr_patches.append(_crop_chw(lr[i], top_lr, left_lr, lr_patch_size))
        depth_patches.append(_crop_chw(depth[i], top_lr, left_lr, lr_patch_size))
        motion_patches.append(_crop_chw(motion[i], top_lr, left_lr, lr_patch_size))
        normal_patches.append(_crop_chw(normals[i], top_lr, left_lr, lr_patch_size))
        canvas_patches.append(_crop_chw(canvas[i], top_lr, left_lr, lr_patch_size))
        target_patches.append(_crop_chw(target[i], top_hr, left_hr, hr_patch_size))

    lr_inputs = torch.cat([
        torch.stack(lr_patches, dim=0),
        torch.stack(depth_patches, dim=0),
        torch.stack(motion_patches, dim=0),
        torch.stack(normal_patches, dim=0),
    ], dim=1)
    return {
        "lr_inputs": lr_inputs.contiguous(),
        "target": torch.stack(target_patches, dim=0).contiguous(),
        "motion": torch.stack(motion_patches, dim=0).contiguous(),
        "canvas_hint": torch.stack(canvas_patches, dim=0).contiguous(),
        "coords": torch.tensor(coords, dtype=torch.long),
    }


def sample_v6_trajectory_patch_batch(
    batch: dict[str, torch.Tensor],
    *,
    hr_patch_size: int,
    scale: int,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Importance-sample one aligned patch window per trajectory item."""
    lr = batch["lr_frame"]
    target = batch["gt_hr_frame"]
    depth = batch["depth"]
    motion = batch["motion"]
    normals = batch["normals"]
    canvas = batch["canvas_hint"]

    if lr.dim() != 5:
        raise ValueError(f"expected trajectory batch (B,T,C,H,W); got {tuple(lr.shape)}")
    if target.shape[-1] < hr_patch_size or target.shape[-2] < hr_patch_size:
        raise ValueError(
            f"target spatial size {tuple(target.shape[-2:])} smaller than "
            f"patch_size={hr_patch_size}"
        )
    if hr_patch_size % scale != 0:
        raise ValueError(f"patch_size={hr_patch_size} must be divisible by scale={scale}")
    lr_patch_size = hr_patch_size // scale

    bsz, traj_len = lr.shape[:2]
    lr_patches = []
    target_patches = []
    depth_patches = []
    motion_patches = []
    normal_patches = []
    canvas_patches = []
    coords: list[tuple[int, int]] = []

    for i in range(bsz):
        # Pick a stable crop for the whole trajectory so canvas continuity and
        # temporal loss are not polluted by per-frame crop jumps.
        top_lr, left_lr = importance_weighted_patch_indices(
            lr[i, 0].detach().float().cpu(),
            patch_size=lr_patch_size,
            num_patches=1,
            importance_ratio=0.7,
            generator=generator,
        )[0]
        top_hr = top_lr * scale
        left_hr = left_lr * scale
        coords.append((top_hr, left_hr))

        lr_frames = []
        target_frames = []
        depth_frames = []
        motion_frames = []
        normal_frames = []
        canvas_frames = []
        for t in range(traj_len):
            lr_frames.append(_crop_chw(lr[i, t], top_lr, left_lr, lr_patch_size))
            depth_frames.append(_crop_chw(depth[i, t], top_lr, left_lr, lr_patch_size))
            motion_frames.append(_crop_chw(motion[i, t], top_lr, left_lr, lr_patch_size))
            normal_frames.append(_crop_chw(normals[i, t], top_lr, left_lr, lr_patch_size))
            canvas_frames.append(_crop_chw(canvas[i, t], top_lr, left_lr, lr_patch_size))
            target_frames.append(_crop_chw(target[i, t], top_hr, left_hr, hr_patch_size))

        lr_patches.append(torch.stack(lr_frames, dim=0))
        depth_patches.append(torch.stack(depth_frames, dim=0))
        motion_patches.append(torch.stack(motion_frames, dim=0))
        normal_patches.append(torch.stack(normal_frames, dim=0))
        canvas_patches.append(torch.stack(canvas_frames, dim=0))
        target_patches.append(torch.stack(target_frames, dim=0))

    lr_inputs = torch.cat([
        torch.stack(lr_patches, dim=0),
        torch.stack(depth_patches, dim=0),
        torch.stack(motion_patches, dim=0),
        torch.stack(normal_patches, dim=0),
    ], dim=2)
    return {
        "lr_inputs": lr_inputs.contiguous(),
        "target": torch.stack(target_patches, dim=0).contiguous(),
        "motion": torch.stack(motion_patches, dim=0).contiguous(),
        "canvas_hint": torch.stack(canvas_patches, dim=0).contiguous(),
        "coords": torch.tensor(coords, dtype=torch.long),
    }


def _next_batch(loader: DataLoader, iterator: Optional[Iterator], step: int) -> tuple[dict, Iterator]:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(step)
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def _autocast_for(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _zero_optimizers(*optimizers: torch.optim.Optimizer) -> None:
    for optim in optimizers:
        optim.zero_grad(set_to_none=True)


def _has_global_nonfinite(tensor: torch.Tensor) -> bool:
    local_bad = int(not bool(torch.isfinite(tensor).all().detach().item()))
    flag = torch.tensor(local_bad, device=tensor.device, dtype=torch.int32)
    if _is_distributed():
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.detach().cpu().item())


def _has_global_nonfinite_grad(
    parameters: list[torch.nn.Parameter],
    *,
    ref: torch.Tensor,
) -> bool:
    local_bad = 0
    for p in parameters:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            local_bad = 1
            break
    flag = torch.tensor(local_bad, device=ref.device, dtype=torch.int32)
    if _is_distributed():
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.detach().cpu().item())


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )


def build_scheduler(
    optim: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> CosineLRWithWarmRestarts:
    return CosineLRWithWarmRestarts(
        optim,
        base_lr=args.base_lr,
        T_0=args.T0,
        T_mult=1.0,
        num_restarts=args.num_restarts,
    )


def scheduler_state_dict(sched: CosineLRWithWarmRestarts) -> dict[str, Any]:
    return {
        "base_lr": sched.base_lr,
        "T_0": sched.T_0,
        "T_mult": sched.T_mult,
        "num_restarts": sched.num_restarts,
        "last_lr": sched.get_last_lr(),
    }


def load_scheduler_state_dict(
    sched: CosineLRWithWarmRestarts,
    state: Optional[dict[str, Any]],
    *,
    step: int,
) -> None:
    if not state:
        sched.step(step)
        return
    sched.base_lr = float(state.get("base_lr", sched.base_lr))
    sched._last_lr = float(state.get("last_lr", sched.get_last_lr()))
    sched.step(step)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _load_rng_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])


def save_checkpoint(
    output_dir: Path,
    step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    sched_g: CosineLRWithWarmRestarts,
    sched_d: CosineLRWithWarmRestarts,
    ema: EMAModel,
    args: argparse.Namespace,
) -> Path:
    ckpt_path = output_dir / f"step-{step:08d}.pt"
    payload: dict[str, Any] = {
        "step": int(step),
        "kind": "v6",
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "generator": _unwrap(generator).state_dict(),
        "discriminator": _unwrap(discriminator).state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "sched_g": scheduler_state_dict(sched_g),
        "sched_d": scheduler_state_dict(sched_d),
        "ema": ema.state_dict(),
        "rng": _rng_state(),
    }
    torch.save(payload, ckpt_path)
    log.info("ckpt -> %s", ckpt_path)
    return ckpt_path


def load_latest_checkpoint(
    output_dir: Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    sched_g: CosineLRWithWarmRestarts,
    sched_d: CosineLRWithWarmRestarts,
    ema: EMAModel,
    device: str,
) -> int:
    if not output_dir.exists():
        return 0
    ckpts = sorted(output_dir.glob("step-*.pt"))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    log.info("auto-resume: loading %s", latest)
    ck = torch.load(latest, map_location=device, weights_only=False)
    g_result = _unwrap(generator).load_state_dict(ck["generator"], strict=False)
    if g_result.missing_keys or g_result.unexpected_keys:
        log.warning(
            "generator checkpoint schema drift: missing=%s unexpected=%s",
            g_result.missing_keys,
            g_result.unexpected_keys,
        )
    d_result = _unwrap(discriminator).load_state_dict(ck["discriminator"], strict=False)
    if d_result.missing_keys or d_result.unexpected_keys:
        log.warning(
            "discriminator checkpoint schema drift: missing=%s unexpected=%s",
            d_result.missing_keys,
            d_result.unexpected_keys,
        )
    if "optim_g" in ck:
        try:
            optim_g.load_state_dict(ck["optim_g"])
        except ValueError as exc:
            log.warning("skipping incompatible generator optimizer state: %s", exc)
    if "optim_d" in ck:
        try:
            optim_d.load_state_dict(ck["optim_d"])
        except ValueError as exc:
            log.warning("skipping incompatible discriminator optimizer state: %s", exc)
    step = int(ck.get("step", 0))
    load_scheduler_state_dict(sched_g, ck.get("sched_g"), step=step)
    load_scheduler_state_dict(sched_d, ck.get("sched_d"), step=step)
    if "ema" in ck:
        try:
            ema.load_state_dict(ck["ema"])
        except (RuntimeError, KeyError, ValueError) as exc:
            log.warning("skipping incompatible EMA state: %s", exc)
    _load_rng_state(ck.get("rng"))
    log.info("resumed at step=%d", step)
    return step


def append_metrics_jsonl(output_dir: Path, row: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def train_step(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    loss_fn: V6CompositeLoss,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    ema: EMAModel,
    batch: dict[str, torch.Tensor],
    *,
    step: int,
    args: argparse.Namespace,
    zero_grad: bool = True,
    step_optim: bool = True,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    """One optimizer step over one trajectory. Runs D only after GAN warmup."""
    device = args.device
    do_gan = step >= args.warmup_steps

    lr_inputs = batch["lr_inputs"]
    target = batch["target"]
    motion = batch["motion"]
    if lr_inputs.dim() == 4:
        lr_inputs = lr_inputs.unsqueeze(1)
        target = target.unsqueeze(1)
        motion = motion.unsqueeze(1)
    if lr_inputs.dim() != 5:
        raise ValueError(f"expected lr_inputs (B,T,C,H,W); got {tuple(lr_inputs.shape)}")

    _unwrap(generator).reset_state(device=torch.device(device))

    if zero_grad:
        optim_g.zero_grad(set_to_none=True)
        optim_d.zero_grad(set_to_none=True)

    _set_requires_grad(discriminator, False)
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_g_loss: Optional[torch.Tensor] = None
    accum_parts: dict[str, float] = {}
    canvas_count_after_frame0 = 0.0
    canvas_count_at_frame1_start = 0.0

    with _autocast_for(device, args.bf16):
        traj_len = int(lr_inputs.shape[1])
        pred_prev: Optional[torch.Tensor] = None
        target_prev: Optional[torch.Tensor] = None
        for frame_idx in range(traj_len):
            module = _unwrap(generator)
            if frame_idx == 1 and hasattr(module, "_canvas_state"):
                canvas = getattr(module, "_canvas_state", None)
                canvas_count_at_frame1_start = float(getattr(canvas, "count", 0) or 0)

            motion_lr = None if frame_idx == 0 else motion[:, frame_idx]
            pred = generator(
                lr_inputs[:, frame_idx],
                motion_lr=motion_lr,
                frame_index=frame_idx,
            )
            if frame_idx == 0 and hasattr(module, "_canvas_state"):
                canvas = getattr(module, "_canvas_state", None)
                canvas_count_after_frame0 = float(getattr(canvas, "count", 0) or 0)

            fake_logits_g = discriminator(pred) if do_gan else None
            frame_loss, parts = loss_fn(
                pred,
                target[:, frame_idx],
                fake_logits=fake_logits_g,
                step=step,
                pred_prev=pred_prev,
                motion_lr=motion_lr,
                scale_factor=float(getattr(_unwrap(generator), "scale", 2)),
                # target_prev enables the paired-warp form of the
                # temporal-consistency loss (audit HIGH-H1 fix). Without it
                # the loss penalizes correct frame-to-frame change.
                target_prev=target_prev,
            )
            total_g_loss = frame_loss if total_g_loss is None else total_g_loss + frame_loss
            for k, v in parts.items():
                accum_parts[k] = accum_parts.get(k, 0.0) + float(v)
            preds.append(pred)
            targets.append(target[:, frame_idx])
            pred_prev = pred
            target_prev = target[:, frame_idx]
    assert total_g_loss is not None
    g_loss = total_g_loss
    if _has_global_nonfinite(g_loss):
        _zero_optimizers(optim_g, optim_d)
        return {"loss_total": float("nan")}
    (g_loss * float(loss_scale)).backward()
    generator_params = [p for p in generator.parameters() if p.requires_grad]
    if _has_global_nonfinite_grad(generator_params, ref=g_loss):
        _zero_optimizers(optim_g, optim_d)
        return {"loss_total": float("nan")}
    if step_optim:
        grad_norm_g = torch.nn.utils.clip_grad_norm_(generator_params, max_norm=1.0)
        if _has_global_nonfinite(grad_norm_g):
            _zero_optimizers(optim_g, optim_d)
            return {"loss_total": float("nan")}
        optim_g.step()

    d_loss = preds[-1].new_zeros(())
    d_fired = False
    if do_gan:
        _set_requires_grad(discriminator, True)
        with _autocast_for(device, args.bf16):
            real_logits = discriminator(torch.cat(targets, dim=0))
            fake_logits_d = discriminator(torch.cat([p.detach() for p in preds], dim=0))
            d_loss = gan_hinge_d_loss(real_logits, fake_logits_d)
        if _has_global_nonfinite(d_loss):
            _zero_optimizers(optim_g, optim_d)
            return {"loss_total": float("nan")}
        (d_loss * float(loss_scale)).backward()
        discriminator_params = [
            p for p in discriminator.parameters()
            if p.requires_grad
        ]
        if _has_global_nonfinite_grad(discriminator_params, ref=d_loss):
            _zero_optimizers(optim_g, optim_d)
            return {"loss_total": float("nan")}
        if step_optim:
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                discriminator_params,
                max_norm=1.0,
            )
            if _has_global_nonfinite(grad_norm_d):
                _zero_optimizers(optim_g, optim_d)
                return {"loss_total": float("nan")}
            optim_d.step()
        d_fired = True
    _set_requires_grad(discriminator, True)

    pruned = 0
    if step_optim:
        ema.update(_unwrap(generator))
        prune_target = _unwrap(generator)
        try:
            pruned = int(prune_target.maybe_prune(step=step))
        except TypeError:
            pruned = int(prune_target.maybe_prune())

    traj_len_f = float(lr_inputs.shape[1])
    parts_avg = {k: v / traj_len_f for k, v in accum_parts.items()}

    return {
        "loss_total": float(accum_parts.get("total", float(g_loss.detach()))),
        "loss_charbonnier": float(parts_avg.get("charbonnier", 0.0)),
        "loss_lpips": float(parts_avg.get("lpips", 0.0)),
        "loss_msvgg": float(parts_avg.get("vgg", 0.0)),
        "loss_wavelet": float(parts_avg.get("wavelet", 0.0)),
        "loss_sobel": float(parts_avg.get("sobel", 0.0)),
        "loss_gan_g": float(parts_avg.get("gan", 0.0)),
        "loss_gan_d": float(d_loss.detach()) if d_fired else 0.0,
        "loss_tc": float(parts_avg.get("temporal", 0.0)),
        "d_step": 1.0 if d_fired else 0.0,
        "n_pruned": float(pruned),
        "canvas_count_after_frame0": canvas_count_after_frame0,
        "canvas_count_at_frame1_start": canvas_count_at_frame1_start,
    }


def _install_synthetic_canvas_state(model: torch.nn.Module, device: torch.device) -> None:
    from oss.sr.v6.model import CanvasState

    token_dim = int(model.cfg.token_dim)
    canvas = CanvasState(
        positions=torch.tensor([[4.0, 4.0], [10.0, 10.0]], device=device),
        scales=torch.ones(2, 2, device=device),
        rotations=torch.zeros(2, device=device),
        opacities=torch.ones(2, device=device),
        colors=torch.randn(2, token_dim, device=device) * 0.01,
        count=2,
    )
    model._canvas_state = canvas
    model.keyframe_mask._cached_mask = torch.ones(2, dtype=torch.bool, device=device)
    model.keyframe_mask._cached_keyframe = 0
    if torch.is_tensor(model._step_count):
        model._step_count.fill_(1)
    else:
        model._step_count = 1


def run_ddp_smoke_probe(
    generator: torch.nn.Module,
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> None:
    """Exercise DDP sync on both empty-canvas and non-empty-canvas branches."""
    if world_size <= 1:
        return

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP smoke probe requires an initialized process group")

    module = _unwrap(generator)
    device = next(module.parameters()).device
    generator.train()

    lr_inputs = torch.randn(
        1,
        int(module.cfg.in_channels),
        16,
        16,
        device=device,
    )
    motion = torch.zeros(1, 2, 16, 16, device=device)

    for p in generator.parameters():
        p.grad = None
    module.reset_state(device=device)
    empty_out = generator(lr_inputs, motion_lr=motion, frame_index=0)
    empty_loss = empty_out.mean()
    empty_loss.backward()

    for p in generator.parameters():
        p.grad = None
    module.reset_state(device=device)
    _install_synthetic_canvas_state(module, device)
    canvas_out = generator(lr_inputs, motion_lr=motion, frame_index=1)
    canvas_loss = canvas_out.mean()
    canvas_loss.backward()

    canvas_grad = module.canvas_to_token.weight.grad
    if canvas_grad is None or not torch.isfinite(canvas_grad).all():
        raise RuntimeError("DDP smoke probe did not exercise canvas_to_token gradients")

    probe = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2.0
    if not torch.isfinite(probe).all() or abs(float(probe.item()) - expected) > 1e-5:
        raise RuntimeError(
            f"DDP all_reduce probe returned {float(probe.item())}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_train_loader(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    scale: int,
) -> DataLoader:
    distributed = world_size > 1
    if args.smoke:
        length = max(args.max_steps * args.grad_accum * max(1, args.batch_size) * 2, 32)
        ds: Dataset = SyntheticV6Dataset(
            length=length,
            hr_size=args.patch_size,
            scale=scale,
            seed=args.seed,
            trajectory_length=args.trajectory_length,
        )
    else:
        if args.tartanair_root is None and args.hypersim_root is None:
            raise ValueError("Non-smoke training requires --tartanair-root and/or --hypersim-root.")
        ds = build_v6_training_dataset(
            tartanair_root=args.tartanair_root,
            hypersim_root=args.hypersim_root,
            held_out_envs=args.held_out_envs,
            held_out_scenes=args.held_out_scenes,
            tartanair_ratio=args.tartanair_mix_ratio,
            hypersim_ratio=args.hypersim_mix_ratio,
            seed=args.seed,
            trajectory_length=args.trajectory_length,
        )
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        shuffle = False
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0 if args.smoke else args.num_workers,
        persistent_workers=(not args.smoke and args.num_workers > 0),
        drop_last=True,
        pin_memory=args.device.startswith("cuda"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = normalize_args(parse_args(argv))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device, rank, world_size = _ddp_init_if_needed(args.device)
    args.device = device
    is_main = _is_main(rank)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rank_seed = args.seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(rank_seed)

    if is_main:
        log.info(
            "v6 trainer starting | device=%s rank=%d world=%d steps=%d smoke=%s "
            "backbone=%s patch=%d batch=%d accum=%d traj=%d lr=%.2e",
            device, rank, world_size, args.max_steps, args.smoke, args.backbone,
            args.patch_size, args.batch_size, args.grad_accum,
            args.trajectory_length, args.base_lr,
        )
        log.info("output_dir=%s", args.output_dir)

    try:
        from oss.sr.v6.model import V6Config, V6Model

        cfg = V6Config(backbone=args.backbone)
        generator = V6Model(cfg).to(device)
        discriminator = UNetDiscriminator().to(device)
        loss_fn = V6CompositeLoss(gan_warmup_until_step=args.warmup_steps).to(device)

        if is_main:
            n_params_g = sum(p.numel() for p in generator.parameters())
            n_params_d = sum(p.numel() for p in discriminator.parameters())
            log.info(
                "models constructed | G params=%d D params=%d",
                n_params_g, n_params_d,
            )

        optim_g = build_optimizer(generator, args)
        optim_d = build_optimizer(discriminator, args)
        sched_g = build_scheduler(optim_g, args)
        sched_d = build_scheduler(optim_d, args)
        ema = EMAModel(generator, decay=0.999)

        generator_for_train: torch.nn.Module = generator
        discriminator_for_train: torch.nn.Module = discriminator
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ["LOCAL_RANK"])
            ddp_kwargs = {
                "device_ids": [local_rank] if device.startswith("cuda") else None,
                "find_unused_parameters": True,
            }
            generator_for_train = DDP(generator, **ddp_kwargs)
            discriminator_for_train = DDP(discriminator, **ddp_kwargs)

        if args.smoke and world_size > 1:
            run_ddp_smoke_probe(
                generator_for_train,
                args=args,
                rank=rank,
                world_size=world_size,
            )
            for p in generator.parameters():
                p.grad = None

        resume_step = load_latest_checkpoint(
            args.output_dir,
            generator,
            discriminator,
            optim_g,
            optim_d,
            sched_g,
            sched_d,
            ema,
            device,
        )

        loader = build_train_loader(args, rank=rank, world_size=world_size, scale=generator.scale)
        iterator: Optional[Iterator] = None
        patch_gen = torch.Generator().manual_seed(rank_seed + 10_000)

        train_start = time.monotonic()
        final_step = resume_step
        last_parts: dict[str, float] = {}

        step = resume_step
        while step < args.max_steps:
            step += 1
            final_step = step
            accum_parts: dict[str, float] = {}
            bad_step = False

            for micro_idx in range(args.grad_accum):
                raw_batch, iterator = _next_batch(loader, iterator, step)
                moved = _move_batch(raw_batch, device)
                patch_batch = sample_v6_patch_batch(
                    moved,
                    hr_patch_size=args.patch_size,
                    scale=generator.scale,
                    generator=patch_gen,
                )
                parts = train_step(
                    generator_for_train,
                    discriminator_for_train,
                    loss_fn,
                    optim_g,
                    optim_d,
                    ema,
                    patch_batch,
                    step=step,
                    args=args,
                    zero_grad=(micro_idx == 0),
                    step_optim=(micro_idx == args.grad_accum - 1),
                    loss_scale=1.0 / float(args.grad_accum),
                )
                for k, v in parts.items():
                    accum_parts[k] = accum_parts.get(k, 0.0) + float(v)
                if not math.isfinite(parts.get("loss_total", float("nan"))):
                    bad_step = True
                    break

            if bad_step:
                if is_main:
                    log.warning(
                        "non-finite loss at step %d; cleared gradients and skipped "
                        "remaining updates/metrics for this step",
                        step,
                    )
                continue

            last_parts = {
                k: v / float(args.grad_accum)
                for k, v in accum_parts.items()
            }
            if not math.isfinite(last_parts.get("loss_total", float("nan"))):
                if is_main:
                    log.warning(
                        "non-finite accumulated loss at step %d; skipped "
                        "optimizer/EMA/metrics for this step: %r",
                        step,
                        last_parts,
                    )
                continue

            sched_g.step(step)
            sched_d.step(step)
            lr_g = sched_g.get_last_lr()
            lr_d = sched_d.get_last_lr()

            if is_main:
                row = {
                    "step": step,
                    "loss_total": last_parts.get("loss_total", 0.0),
                    "loss_charbonnier": last_parts.get("loss_charbonnier", 0.0),
                    "loss_lpips": last_parts.get("loss_lpips", 0.0),
                    "loss_msvgg": last_parts.get("loss_msvgg", 0.0),
                    "loss_wavelet": last_parts.get("loss_wavelet", 0.0),
                    "loss_sobel": last_parts.get("loss_sobel", 0.0),
                    "loss_gan_g": last_parts.get("loss_gan_g", 0.0),
                    "loss_gan_d": last_parts.get("loss_gan_d", 0.0),
                    "loss_tc": last_parts.get("loss_tc", 0.0),
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "ema_decay": ema.decay,
                }
                append_metrics_jsonl(args.output_dir, row)
                if step % args.log_every == 0 or step == 1 or args.smoke:
                    log.info(
                        "step=%d loss=%.4f char=%.4f lpips=%.4f gan_g=%.4f gan_d=%.4f",
                        step,
                        row["loss_total"],
                        row["loss_charbonnier"],
                        row["loss_lpips"],
                        row["loss_gan_g"],
                        row["loss_gan_d"],
                    )

                early_ckpt = (
                    args.first_ckpt_step > 0
                    and step == args.first_ckpt_step
                    and step < args.max_steps
                    and step % args.ckpt_every != 0
                )
                if early_ckpt or step % args.ckpt_every == 0 or step == args.max_steps:
                    save_checkpoint(
                        args.output_dir,
                        step,
                        generator,
                        discriminator,
                        optim_g,
                        optim_d,
                        sched_g,
                        sched_d,
                        ema,
                        args,
                    )

        if is_main and final_step > 0:
            final_ckpt = args.output_dir / f"step-{final_step:08d}.pt"
            if not final_ckpt.exists():
                save_checkpoint(
                    args.output_dir,
                    final_step,
                    generator,
                    discriminator,
                    optim_g,
                    optim_d,
                    sched_g,
                    sched_d,
                    ema,
                    args,
                )

        elapsed = time.monotonic() - train_start
        if is_main:
            print(
                f"v6 training: device={device} world_size={world_size} smoke={args.smoke}",
                flush=True,
            )
            print(f"final_step={final_step} elapsed={elapsed:.1f}s", flush=True)
            print(
                f"final_loss={last_parts.get('loss_total', float('nan')):.6f}",
                flush=True,
            )
            print(f"checkpoint -> {args.output_dir}/step-{final_step:08d}.pt", flush=True)
            print("done.", flush=True)

        _ddp_cleanup()
        return 0
    except Exception as e:
        if is_main:
            log.exception("v6 trainer failed cleanly: %s", e)
        _ddp_cleanup()
        return 10


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""v6 training script. Consumes ``oss.sr.v6`` modules.

Mirrors the v5 trainer's structure (auto-resume, metrics dump, dashboard
compat) but with v6's loss recipe, EMA, cosine LR + warm restarts, and
mixed TartanAir + Hypersim dataset.

Per the v6 architecture memo (``docs/superpowers/experiments/
2026-05-05-v6-architecture-canonical.md``) section 6:

    Optimizer:    AdamW, beta=(0.9, 0.99), wd=1e-4
    LR:           2e-4 cosine + 3 warm restarts, T_0=50K, T_mult=1
    Precision:    bf16
    Effective batch: 16 (batch=4, accum=4) for HAT-Base teacher
    Patch size:   256^2 (Heavy / HAT-Base) ; 192^2 (Standard) ; 128^2 (Pico)
    Patch sampling: 70% importance (variance-weighted) + 30% uniform
    EMA:          beta=0.999 (teacher only)
    Steps:        300K (teacher / Heavy)
    Data mix:     TartanAir 60% + Hypersim 30% + held-out 10%

This file is the entry point; the V6Model glue currently raises
NotImplementedError (see ``oss/sr/v6/model.py``). ``--smoke`` reaches the
V6Model construction and exits with the not-implemented message instead
of crashing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Allow ``python scripts/sr_train_v6.py`` to import ``oss.*`` without
# ``pip install -e .`` first. Mirrors sr_train_gaussian_temporal.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from oss.sr.v6.dataset import build_v6_training_dataset  # noqa: E402

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

    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--warmup-steps", type=int, default=20_000,
                   help="GAN warmup until this step (pixel-only before).")
    p.add_argument("--T0", type=int, default=50_000,
                   help="Cosine warm-restart period (T_mult=1).")
    p.add_argument("--num-restarts", type=int, default=3)

    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=4)

    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--backbone", choices=("hat-tiny", "hat-small", "hat-l"),
                   default="hat-l")
    p.add_argument("--warm-start", type=Path, default=None,
                   help="HAT-L SA1B warm-start ckpt (from GSASR) — optional.")

    p.add_argument("--ckpt-every", type=int, default=5_000)

    p.add_argument("--hypersim-mix-ratio", type=float, default=0.333)
    p.add_argument("--tartanair-mix-ratio", type=float, default=0.667)

    p.add_argument("--smoke", action="store_true",
                   help="Wire through to V6Model construction; expect "
                        "NotImplementedError until V6Model lands.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_NOT_IMPLEMENTED_PREFIX = "V6Model not yet implemented:"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device, rank, world_size = _ddp_init_if_needed(args.device)
    if _is_main(rank):
        log.info("v6 trainer starting | device=%s rank=%d world=%d", device, rank, world_size)
        log.info("output_dir=%s", args.output_dir)

    # Step 2 — datasets (skipped in smoke if no roots provided).
    train_ds: Any = None
    if not args.smoke and (args.tartanair_root is not None or args.hypersim_root is not None):
        if _is_main(rank):
            log.info(
                "building v6 dataset: tartanair=%s hypersim=%s ratios=(%.3f, %.3f)",
                args.tartanair_root, args.hypersim_root,
                args.tartanair_mix_ratio, args.hypersim_mix_ratio,
            )
        train_ds = build_v6_training_dataset(
            tartanair_root=args.tartanair_root,
            hypersim_root=args.hypersim_root,
            held_out_envs=args.held_out_envs,
            held_out_scenes=args.held_out_scenes,
            tartanair_ratio=args.tartanair_mix_ratio,
            hypersim_ratio=args.hypersim_mix_ratio,
            seed=args.seed,
        )
        if _is_main(rank):
            log.info("v6 mixed dataset length: %d", len(train_ds))

    # Step 3 — V6Model (orchestrator stub raises NotImplementedError).
    try:
        from oss.sr.v6.model import V6Model
        _model = V6Model(
            backbone=args.backbone,
            warm_start=args.warm_start,
            patch_size=args.patch_size,
        )
    except NotImplementedError as e:
        msg = f"{_NOT_IMPLEMENTED_PREFIX} {e}"
        if _is_main(rank):
            log.warning(msg)
            print(msg)
        # Persist a stub metrics.json so the dashboard parser still reads
        # something coherent.
        metrics_path = args.output_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path.write_text(json.dumps({"status": "v6-model-stub", "step": 0}))
        _ddp_cleanup()
        return 0

    # ---- Steps 4-10: full training loop, will be implemented when V6Model
    # lands. Skeleton only below; not exercised by the current stub path.
    raise RuntimeError(  # pragma: no cover
        "Reached the v6 training-loop section, but it is not yet implemented. "
        "This branch should be unreachable while V6Model raises "
        "NotImplementedError on construction."
    )


if __name__ == "__main__":
    sys.exit(main())

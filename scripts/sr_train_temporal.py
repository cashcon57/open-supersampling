#!/usr/bin/env python
"""v5 pixel-temporal SR training entry.

Three-phase schedule (default):
    1. Steps  0..10000 — backbone frozen; head + gate only; appearance loss only.
    2. Steps 10000..60000 — backbone unfrozen at LR*0.1; full loss with
       temporal-consistency at lambda=0.05.
    3. Steps 60000..80000 — Sintel-only fine-tune at LR*0.01.

Writes ``metrics.json`` (keyed by step) and an initially empty
``score_log.json``. Training progress lives in ``metrics.json``; real
held-out eval rows are written later by ``scripts/sr_temporal_held_out.py``.
Auto-resumes from the latest checkpoint in ``--output-dir`` if any.

Smoke mode (``--smoke``):
    Runs 5 CPU steps on synthetic random tensors. Used in CI / pre-launch.

Mirrors ``oss/gaussian/train/train.py`` for auto-resume + metrics-dump
patterns: rolling write of ``metrics.json`` and ``score_log.json`` at every
checkpoint; ``score_log.json`` is preserved for held-out eval rows.
Auto-resume picks the latest ``step-XXXXX.pt`` from
``--output-dir`` and rehydrates optimizer state + previously logged metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow ``python scripts/sr_train_temporal.py`` to import ``oss.*`` when the
# package isn't installed into the active interpreter (e.g. tests invoke us
# via ``sys.executable`` on a system Python). Inserting the repo root keeps
# the script self-contained and avoids requiring ``pip install -e .`` first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from oss.sr.temporal import (
    TemporalSRModel,
    make_first_frame_prev_hr,
)
from oss.train.losses import temporal_consistency_loss

log = logging.getLogger("oss.sr.temporal.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write checkpoints + metrics.json + score_log.json")
    p.add_argument("--warm-start", type=Path, default=None,
                   help="Optional v4 checkpoint (.pt) to warm-start the backbone from.")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="Root of the extracted TartanAir dataset (Phase 1+2).")
    p.add_argument("--sintel-root", type=Path, default=None,
                   help="Root of the Sintel dataset (Phase 2+3).")
    p.add_argument("--max-steps", type=int, default=80_000)
    p.add_argument("--warmup-steps", type=int, default=10_000,
                   help="End of Phase 1 (backbone frozen, appearance loss only).")
    p.add_argument("--joint-end", type=int, default=60_000,
                   help="End of Phase 2 (joint TartanAir+Sintel with temporal loss).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Synthetic random tensors, no datasets. CI / pre-launch sanity.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lpips-weight", type=float, default=0.1,
                   help="Perceptual loss weight (Phase 2/3).")
    p.add_argument("--tc-weight", type=float, default=0.05,
                   help="Temporal-consistency loss weight (Phase 2/3).")
    p.add_argument("--ssim-weight", type=float, default=0.1)
    p.add_argument("--tier", default="standard",
                   choices=["pico", "lite", "standard"])
    p.add_argument("--backbone-kind", default="simple",
                   choices=["simple", "rrdb"])
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase schedule
# ---------------------------------------------------------------------------


def phase_for_step(step: int, warmup_steps: int, joint_end: int) -> int:
    """Return 1, 2, or 3 for the current 3-phase schedule."""
    if step <= warmup_steps:
        return 1
    if step <= joint_end:
        return 2
    return 3


def lr_multiplier_for_phase(phase: int) -> float:
    """LR scaling per phase: Phase 1 = 1.0, Phase 2 = 0.1, Phase 3 = 0.01."""
    if phase == 1:
        return 1.0
    if phase == 2:
        return 0.1
    return 0.01


def apply_phase(
    model: TemporalSRModel,
    optim: torch.optim.Optimizer,
    base_lr: float,
    prev_phase: int,
    cur_phase: int,
) -> None:
    """Apply the per-phase backbone-freeze + LR-scale on transition."""
    if cur_phase == prev_phase:
        return
    # Phase 1: freeze backbone; Phase 2/3: unfrozen.
    model.freeze_backbone(freeze=(cur_phase == 1))
    mult = lr_multiplier_for_phase(cur_phase)
    for pg in optim.param_groups:
        pg["lr"] = base_lr * mult
    log.info("phase transition: %d -> %d  (lr=%.2e, backbone_frozen=%s)",
             prev_phase, cur_phase, base_lr * mult, cur_phase == 1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


_LPIPS_FN = None
_LPIPS_TRIED = False


def _get_lpips_fn(device: str):
    global _LPIPS_FN, _LPIPS_TRIED
    if _LPIPS_TRIED:
        return _LPIPS_FN
    _LPIPS_TRIED = True
    try:
        import lpips  # type: ignore[import-not-found]
        _LPIPS_FN = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in _LPIPS_FN.parameters():
            p.requires_grad_(False)
    except Exception:
        _LPIPS_FN = None
    return _LPIPS_FN


def _pooled_l1_ssim_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cheap dependency-free SSIM stand-in: pooled-L1 of luminance."""
    pl = pred.mean(dim=1, keepdim=True)
    tl = target.mean(dim=1, keepdim=True)
    return F.l1_loss(F.avg_pool2d(pl, 8, 8), F.avg_pool2d(tl, 8, 8))


def appearance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    w_l1: float = 1.0,
    w_ssim: float = 0.1,
    w_lpips: float = 0.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 + (1 - SSIM-ish) [+ LPIPS-VGG]."""
    l1 = F.l1_loss(pred, target)
    try:
        from pytorch_msssim import ssim as _ssim_fn  # type: ignore[import-not-found]
        ssim_val = _ssim_fn(
            pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0),
            data_range=1.0, size_average=True,
        )
        loss = w_l1 * l1 + w_ssim * (1.0 - ssim_val)
        parts: dict[str, float] = {
            "l1": float(l1.item()),
            "ssim": float(ssim_val.item()),
        }
    except Exception:
        proxy = _pooled_l1_ssim_proxy(pred, target)
        loss = w_l1 * l1 + w_ssim * proxy
        parts = {"l1": float(l1.item()), "pooled_l1": float(proxy.item())}

    if w_lpips > 0:
        fn = _get_lpips_fn(device)
        if fn is not None:
            p = pred.clamp(0.0, 1.0) * 2 - 1
            t = target.clamp(0.0, 1.0) * 2 - 1
            lp = fn(p, t).mean()
            loss = loss + w_lpips * lp
            parts["lpips"] = float(lp.item())
    return loss, parts


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_datasets(args: argparse.Namespace):
    """Build SequentialPairDataset loaders for TartanAir + Sintel.

    Returns ``(tartan_loader_or_None, sintel_loader_or_None)``. Either may be
    None if the corresponding root flag was not provided. The training loop
    selects which to draw from each phase.
    """
    from torch.utils.data import DataLoader

    from oss.gaussian.data import (
        SintelGaussianDataset,
        TartanAirGaussianDataset,
    )
    from oss.sr.temporal import (
        SequentialPairDataset, adapt_sintel, adapt_tartanair, default_collate_pair,
    )

    tartan_loader = None
    sintel_loader = None

    if args.tartanair_root is not None:
        ds_t = TartanAirGaussianDataset(root=args.tartanair_root, scale=2.0)
        ds_t = adapt_tartanair(ds_t)
        pair_t = SequentialPairDataset(ds_t)
        tartan_loader = DataLoader(
            pair_t, batch_size=args.batch_size, shuffle=True,
            num_workers=2, collate_fn=default_collate_pair, drop_last=True,
        )
    if args.sintel_root is not None:
        ds_s = SintelGaussianDataset(root=args.sintel_root, scale=2.0, pass_name="clean")
        ds_s = adapt_sintel(ds_s)
        pair_s = SequentialPairDataset(ds_s)
        sintel_loader = DataLoader(
            pair_s, batch_size=args.batch_size, shuffle=True,
            num_workers=2, collate_fn=default_collate_pair, drop_last=True,
        )
    return tartan_loader, sintel_loader


def select_loader_for_phase(
    phase: int, tartan_loader, sintel_loader,
):
    """Return the dataset to draw from given the current phase."""
    if phase == 3:
        return sintel_loader or tartan_loader
    return tartan_loader or sintel_loader


# ---------------------------------------------------------------------------
# Synthetic batch (smoke mode)
# ---------------------------------------------------------------------------


def synthetic_pair_batch(
    batch_size: int, height: int, width: int, scale: int, device: str,
) -> dict[str, torch.Tensor]:
    """Synthetic frame-pair batch matching the SequentialPairDataset collate.

    Used only by --smoke. Exercises the full training pipeline on random
    tensors with no dataset on disk.
    """
    g = torch.Generator(device=device).manual_seed(int(time.time()) & 0xFFFF)
    H_hr, W_hr = height * scale, width * scale

    def _frame() -> dict[str, torch.Tensor]:
        return {
            "lr": torch.empty(
                (batch_size, 3, height, width), device=device
            ).uniform_(0, 1, generator=g),
            "depth": torch.empty(
                (batch_size, 1, height, width), device=device
            ).uniform_(0, 1, generator=g),
            "motion": torch.empty(
                (batch_size, 2, height, width), device=device
            ).uniform_(-0.5, 0.5, generator=g),
            "normals": torch.empty(
                (batch_size, 3, height, width), device=device
            ).uniform_(-1, 1, generator=g),
            "canvas": torch.empty(
                (batch_size, 3, height, width), device=device
            ).uniform_(0, 1, generator=g),
            "gt_hr": torch.empty(
                (batch_size, 3, H_hr, W_hr), device=device
            ).uniform_(0, 1, generator=g),
        }

    f_t = _frame()
    f_tp1 = _frame()
    out: dict[str, torch.Tensor] = {}
    for k, v in f_t.items():
        out[f"t_{k}"] = v
    for k, v in f_tp1.items():
        out[f"tp1_{k}"] = v
    out["is_first_in_seq"] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return out


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def _make_12ch_input(lr: torch.Tensor, depth: torch.Tensor,
                     motion: torch.Tensor, normals: torch.Tensor,
                     canvas: torch.Tensor) -> torch.Tensor:
    """Concatenate the 12-channel network input the v4 backbone expects."""
    return torch.cat([lr, depth, motion, normals, canvas], dim=1)


def train_step(
    model: TemporalSRModel,
    batch: dict[str, torch.Tensor],
    optim: torch.optim.Optimizer,
    *,
    phase: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """One optimizer step on a frame-pair batch.

    Phase 1: appearance-only loss on (out_t and out_tp1) vs GT; backbone frozen.
    Phase 2/3: appearance + temporal-consistency.
    """
    device = args.device

    # Move + assemble.
    t_lr = batch["t_lr"].to(device)
    t_depth = batch["t_depth"].to(device)
    t_motion = batch["t_motion"].to(device)
    t_normals = batch["t_normals"].to(device)
    t_canvas = batch["t_canvas"].to(device)
    t_gt = batch["t_gt_hr"].to(device)

    p_lr = batch["tp1_lr"].to(device)
    p_depth = batch["tp1_depth"].to(device)
    p_motion = batch["tp1_motion"].to(device)
    p_normals = batch["tp1_normals"].to(device)
    p_canvas = batch["tp1_canvas"].to(device)
    p_gt = batch["tp1_gt_hr"].to(device)

    H_hr = t_gt.shape[-2]
    W_hr = t_gt.shape[-1]

    # Frame t: cold start with bilinear-up of LR as prev_hr (matches the
    # deployed inference engine on first frame).
    prev_hr_t = make_first_frame_prev_hr(t_lr, scale=model.scale)
    # Upsample HR depth from the LR depth (same trick used in the loss test).
    depth_hr_curr_t = F.interpolate(
        t_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
    )
    depth_hr_prev_t = depth_hr_curr_t  # cold start

    x_t = _make_12ch_input(t_lr, t_depth, t_motion, t_normals, t_canvas)
    out_t = model(
        lr_inputs=x_t, prev_hr=prev_hr_t,
        depth_hr_curr=depth_hr_curr_t, depth_hr_prev=depth_hr_prev_t,
        motion_lr=t_motion,
    )

    # Frame t+1: prev_hr = out_t.detach() (recurrent rollout).
    # IMPORTANT: motion fed to the temporal head is the flow that aligns
    # prev_hr (=out_t) with the CURRENT frame (t+1). TartanAir/Sintel store
    # forward flow ``t -> t+1`` AT frame t (``t_motion``), so for the t+1
    # render we must pass ``t_motion``, NOT ``p_motion`` (which is the flow
    # for t+1 -> t+2 alignment, used on the NEXT step).
    depth_hr_curr_tp1 = F.interpolate(
        p_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
    )
    # x_tp1 packs the t+1 frame's own G-buffers; the motion channel inside
    # the 12-ch stack is informational for the backbone (G-buffer hint), but
    # the EXPLICIT ``motion_lr`` arg below is the one that drives the warp.
    x_tp1 = _make_12ch_input(p_lr, p_depth, p_motion, p_normals, p_canvas)
    out_tp1 = model(
        lr_inputs=x_tp1, prev_hr=out_t.detach(),
        depth_hr_curr=depth_hr_curr_tp1, depth_hr_prev=depth_hr_curr_t,
        motion_lr=t_motion,  # forward flow t -> t+1, sampled at t+1 grid (small-motion approx)
    )

    # Appearance loss on both frames.
    w_lpips_eff = args.lpips_weight if phase != 1 else 0.0
    loss_t, parts_t = appearance_loss(
        out_t, t_gt,
        w_l1=1.0, w_ssim=args.ssim_weight, w_lpips=w_lpips_eff, device=device,
    )
    loss_tp1, parts_tp1 = appearance_loss(
        out_tp1, p_gt,
        w_l1=1.0, w_ssim=args.ssim_weight, w_lpips=w_lpips_eff, device=device,
    )
    loss = loss_t + loss_tp1
    parts = {f"t_{k}": v for k, v in parts_t.items()}
    parts.update({f"tp1_{k}": v for k, v in parts_tp1.items()})

    # Temporal consistency (Phase 2/3 only).
    # The consistency loss warps ``out_t`` by the t -> t+1 flow and compares
    # to ``out_tp1``. Same convention as above: forward flow lives at frame t,
    # so use ``t_motion``.
    if phase != 1:
        tc = temporal_consistency_loss(
            out_tp1, out_t, t_motion, scale_factor=float(model.scale),
        )
        loss = loss + args.tc_weight * tc
        parts["tc"] = float(tc.item())

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optim.step()

    parts["loss"] = float(loss.item())
    parts["phase"] = float(phase)
    return parts


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirror oss/gaussian/train/train.py auto-resume pattern)
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: TemporalSRModel,
    optim: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    """Save a rolling training checkpoint.

    Schema deliberately matches the v4 SR checkpoint shape used by
    ``oss/gaussian/train/train.py``: ``step``, ``args`` (str-coerced Paths),
    ``optim``, plus a kind-specific model state dict — here ``temporal_model``
    so the loader can disambiguate from a plain v4 ``sr_model`` checkpoint.
    """
    ckpt_path = output_dir / f"step-{step:08d}.pt"
    payload: dict[str, Any] = {
        "step": step,
        "kind": "temporal",
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "temporal_model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(payload, ckpt_path)
    log.info("ckpt -> %s", ckpt_path)
    return ckpt_path


def load_latest_checkpoint(
    output_dir: Path,
    model: TemporalSRModel,
    optim: torch.optim.Optimizer,
    device: str,
) -> tuple[int, list[dict], list[dict]]:
    """Auto-resume from the most recent ``step-*.pt`` in ``output_dir``.

    Returns ``(resume_step, metrics_log, score_log)``. If no checkpoint
    exists, returns ``(0, [], [])``.

    Mirrors the gaussian path: also rehydrates ``metrics.json`` + ``score_log.json``
    so dashboards see a continuous history across restarts.
    """
    if not output_dir.exists():
        return 0, [], []
    ckpts = sorted(output_dir.glob("step-*.pt"))
    if not ckpts:
        return 0, [], []
    latest = ckpts[-1]
    log.info("auto-resume: loading %s", latest)
    ck = torch.load(latest, map_location=device, weights_only=False)
    if "temporal_model" in ck:
        model.load_state_dict(ck["temporal_model"])
    elif "sr_model" in ck:
        # Older checkpoint: just the v4 backbone — load into model.backbone.
        model.backbone.load_state_dict(ck["sr_model"])
    if "optim" in ck:
        try:
            optim.load_state_dict(ck["optim"])
        except (ValueError, KeyError) as e:
            log.warning("optim state_dict mismatch on resume; fresh optim: %s", e)
    resume_step = int(ck.get("step", 0))

    metrics_log: list[dict] = []
    score_log: list[dict] = []
    mp = output_dir / "metrics.json"
    if mp.exists():
        try:
            with mp.open() as f:
                saved = json.load(f)
            metrics_log = saved.get("train", [])
            score_log = saved.get("score", [])
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not parse metrics.json on resume: %s", e)
    sp = output_dir / "score_log.json"
    if sp.exists() and not score_log:
        try:
            with sp.open() as f:
                score_log = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not parse score_log.json on resume: %s", e)
    log.info("resumed at step=%d (metrics=%d, score=%d)",
             resume_step, len(metrics_log), len(score_log))
    return resume_step, metrics_log, score_log


def dump_metrics(
    output_dir: Path,
    metrics_log: list[dict],
    score_log: list[dict],
) -> None:
    """Write rolling metrics.json + score_log.json (dashboard-compatible)."""
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"train": metrics_log, "score": score_log}, f, indent=2)
    score_path = output_dir / "score_log.json"
    with score_path.open("w") as f:
        json.dump(score_log, f, indent=2)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def build_optimizer(model: TemporalSRModel, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


def build_scheduler(optim: torch.optim.Optimizer):
    """Per-phase LR is set imperatively via ``apply_phase``; no torch scheduler.

    Returning ``None`` is intentional — phase transitions are sparse (2 over
    the whole run) and trivial to apply by mutating ``param_groups``.
    """
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    log.info(
        "v5-pixel-temporal: device=%s tier=%s steps=%d batch=%d smoke=%s "
        "warmup=%d joint_end=%d lr=%.2e",
        args.device, args.tier, args.max_steps, args.batch_size, args.smoke,
        args.warmup_steps, args.joint_end, args.lr,
    )

    # Build model.
    if args.warm_start is not None and args.warm_start.exists():
        log.info("warm-start backbone from %s", args.warm_start)
        model = TemporalSRModel.load_v4_warm_start(
            args.warm_start, in_channels=12, scale=2, device=args.device,
        )
    else:
        model = TemporalSRModel(
            in_channels=12, scale=2, tier=args.tier, backbone_kind=args.backbone_kind,
        )
    model.to(args.device)
    log.info("model params: total=%d", sum(p.numel() for p in model.parameters()))

    optim = build_optimizer(model, lr=args.lr)
    _ = build_scheduler(optim)  # not used; phase transitions handle LR

    # Auto-resume.
    resume_step, metrics_log, score_log = load_latest_checkpoint(
        args.output_dir, model, optim, args.device,
    )

    # Initial phase application.
    cur_phase = phase_for_step(
        max(resume_step, 0), args.warmup_steps, args.joint_end,
    )
    apply_phase(model, optim, args.lr, prev_phase=-1, cur_phase=cur_phase)

    train_start = time.monotonic()
    final_step = resume_step

    # Build datasets (skip in smoke mode).
    tartan_loader = sintel_loader = None
    tartan_iter = sintel_iter = None
    if not args.smoke:
        tartan_loader, sintel_loader = build_datasets(args)
        if tartan_loader is None and sintel_loader is None:
            log.error("Non-smoke training requires --tartanair-root and/or --sintel-root.")
            return 2

    step = resume_step
    parts: dict[str, float] = {}
    while step < args.max_steps:
        step += 1
        final_step = step

        new_phase = phase_for_step(step, args.warmup_steps, args.joint_end)
        if new_phase != cur_phase:
            apply_phase(model, optim, args.lr, prev_phase=cur_phase, cur_phase=new_phase)
            cur_phase = new_phase

        # Pull a batch.
        if args.smoke:
            batch = synthetic_pair_batch(
                batch_size=max(1, args.batch_size),
                height=16, width=16, scale=model.scale, device=args.device,
            )
        else:
            loader = select_loader_for_phase(cur_phase, tartan_loader, sintel_loader)
            if loader is None:
                log.error("No loader available for phase %d", cur_phase)
                return 3
            if cur_phase == 2 and tartan_loader is not None and sintel_loader is not None:
                if step % 2 == 0:
                    if sintel_iter is None:
                        sintel_iter = iter(sintel_loader)
                    try:
                        batch = next(sintel_iter)
                    except StopIteration:
                        sintel_iter = iter(sintel_loader)
                        batch = next(sintel_iter)
                else:
                    if tartan_iter is None:
                        tartan_iter = iter(tartan_loader)
                    try:
                        batch = next(tartan_iter)
                    except StopIteration:
                        tartan_iter = iter(tartan_loader)
                        batch = next(tartan_iter)
            else:
                if loader is tartan_loader:
                    if tartan_iter is None:
                        tartan_iter = iter(tartan_loader)
                    try:
                        batch = next(tartan_iter)
                    except StopIteration:
                        tartan_iter = iter(tartan_loader)
                        batch = next(tartan_iter)
                else:
                    if sintel_iter is None:
                        sintel_iter = iter(sintel_loader)
                    try:
                        batch = next(sintel_iter)
                    except StopIteration:
                        sintel_iter = iter(sintel_loader)
                        batch = next(sintel_iter)

        parts = train_step(model, batch, optim, phase=cur_phase, args=args)
        if not math.isfinite(parts["loss"]):
            log.error("non-finite loss at step %d: %r", step, parts)
            return 4

        # Periodic logging.
        if step % args.log_every == 0 or step == 1 or args.smoke:
            row = {"step": step, **parts}
            metrics_log.append(row)
            log.info(
                "step=%d phase=%d loss=%.4f t_l1=%.4f tp1_l1=%.4f%s",
                step, cur_phase, row["loss"], row.get("t_l1", float("nan")),
                row.get("tp1_l1", float("nan")),
                f" tc={row['tc']:.4f}" if "tc" in row else "",
            )

        # Periodic checkpoint + rolling metrics dump.
        # Per Codex finding: do NOT append synthetic eval rows to score_log
        # during training. The dashboard's eval cards / margin lines treat any
        # row in score_log as a real held-out eval — emitting train-loss-derived
        # rows with bicubic=None makes JS coerce null→0, showing a misleading
        # positive PSNR margin before the held-out script has run. Training
        # progress lives in metrics.json train rows; score_log.json stays empty
        # until scripts/sr_temporal_held_out.py populates it.
        if step % args.ckpt_every == 0 or step == args.max_steps or args.smoke:
            save_checkpoint(args.output_dir, step, model, optim, args)
            dump_metrics(args.output_dir, metrics_log, score_log)

    # Final dump (idempotent).
    if final_step > 0:
        save_checkpoint(args.output_dir, final_step, model, optim, args)
        dump_metrics(args.output_dir, metrics_log, score_log)

    elapsed = time.monotonic() - train_start
    final_loss = parts.get("loss", float("nan"))
    print(f"v5-pixel-temporal training: device={args.device} smoke={args.smoke}")
    print(f"final_step={final_step} elapsed={elapsed:.1f}s")
    print(f"final_loss={final_loss:.6f} phase={cur_phase}")
    print(f"checkpoint -> {args.output_dir}/step-{final_step:08d}.pt")
    print("done.")
    return 0


def _approx_psnr_from_l1(l1) -> float:
    """Cheap PSNR proxy from L1 (NOT a real metric). Just for curve shape on
    the dashboard until the held-out script (Task 8) writes real PSNR.
    """
    if l1 is None or l1 <= 0:
        return float("nan")
    return float(-20.0 * math.log10(max(float(l1), 1e-6)))


if __name__ == "__main__":
    sys.exit(main())

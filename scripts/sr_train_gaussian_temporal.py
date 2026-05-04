#!/usr/bin/env python
"""v5 gaussian-temporal SR training entry.

Four-phase schedule (default):
    1. Steps 0..20000      — single-frame fitter: encoder + initial densification
                             + raster only. No transformer use, no temporal loss.
                             ``prev_field`` is always None (treats every frame
                             as a first frame). L1 + SSIM appearance loss only.
    2. Steps 20000..50000  — temporal warmup. Warped prev-field carried across
                             steps; transformer active (2 effective layers per
                             plan — implemented as full-architecture forward
                             with the temporal loss/reg DISABLED so only
                             appearance gradients flow). Encoder frozen.
    3. Steps 50000..120000 — joint training. Encoder unfrozen, full 4-layer
                             transformer, full loss including temporal-consistency
                             + regularization, densification active.
    4. Steps 120000..140000 — Sintel-only fine-tune at LR*0.01.

Writes ``metrics.json`` (keyed by step) and ``score_log.json`` (rolling list of
{step, ..., model_psnr_mean, bicubic_psnr_mean, model_lpips_mean, bicubic_lpips_mean})
compatible with ``scripts/training_dashboard.py``. Auto-resumes from the
latest checkpoint in ``--output-dir`` if any. Mirrors the auto-resume +
metrics-dump patterns of ``scripts/sr_train_temporal.py``.

BPTT detach: each step's ``prev_field`` is detached before being fed to the
next step (training-graph length = 1 frame). The temporal-consistency loss
provides the only gradient path that bridges t and t+1 (via ``out_prev``).

Motion convention: when training the t -> t+1 transition, we feed the model
the flow stored AT frame t (forward flow t -> t+1). This matches the pixel
training fix and the convention used by ``temporal_consistency_loss``.

Smoke mode (``--smoke``):
    Runs end-to-end on a synthetic moving-rectangle batch on CPU. The model
    expects B=1, so smoke uses ``batch_size=1`` regardless of CLI flag.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Allow ``python scripts/sr_train_gaussian_temporal.py`` to import ``oss.*``
# without ``pip install -e .`` first. Mirrors sr_train_temporal.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from oss.sr.gaussian_temporal import (
    GaussianField,
    GaussianTemporalSRModel,
    gaussian_regularization_loss,
)
from oss.train.losses import temporal_consistency_loss

log = logging.getLogger("oss.sr.gaussian_temporal.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for checkpoints + metrics.json + score_log.json")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="Root of the extracted TartanAir dataset (Phase 1+2+3).")
    p.add_argument("--sintel-root", type=Path, default=None,
                   help="Root of the Sintel dataset (Phase 2+3+4).")
    p.add_argument("--max-steps", type=int, default=140_000)
    p.add_argument("--phase1-end", type=int, default=20_000,
                   help="End of Phase 1 (single-frame fitter).")
    p.add_argument("--phase2-end", type=int, default=50_000,
                   help="End of Phase 2 (temporal warmup, encoder frozen).")
    p.add_argument("--phase3-end", type=int, default=120_000,
                   help="End of Phase 3 (joint full training).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Effective per-step batch (model expects B=1; values >1 "
                        "are gradient-accumulated via per-sample microbatches).")
    p.add_argument("--window", type=int, default=5,
                   help="Trajectory window length (# consecutive frames).")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Synthetic moving-rectangle, no datasets. CI / pre-launch.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ssim-weight", type=float, default=0.1)
    p.add_argument("--tc-weight", type=float, default=0.05,
                   help="Temporal-consistency loss weight (Phase 3+).")
    p.add_argument("--reg-weight", type=float, default=0.01,
                   help="Gaussian regularization loss weight (Phase 3+).")
    p.add_argument("--max-count", type=int, default=16384,
                   help="Max # alive Gaussians in the field.")
    p.add_argument("--max-area", type=float, default=64.0,
                   help="Hinge knee for covariance area regularization term.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase schedule
# ---------------------------------------------------------------------------


def phase_for_step(step: int, p1: int, p2: int, p3: int) -> int:
    """Return 1, 2, 3, or 4 for the current 4-phase schedule."""
    if step <= p1:
        return 1
    if step <= p2:
        return 2
    if step <= p3:
        return 3
    return 4


def lr_multiplier_for_phase(phase: int) -> float:
    """LR scaling per phase. Phase 1/2/3 = 1.0; Phase 4 = 0.01."""
    if phase == 4:
        return 0.01
    return 1.0


def apply_phase(
    model: GaussianTemporalSRModel,
    optim: torch.optim.Optimizer,
    base_lr: float,
    prev_phase: int,
    cur_phase: int,
) -> None:
    """Apply per-phase param-freeze + LR-scale on transition."""
    if cur_phase == prev_phase:
        return
    # Encoder frozen during Phase 2 only.
    encoder_frozen = (cur_phase == 2)
    for p in model.encoder.parameters():
        p.requires_grad_(not encoder_frozen)
    mult = lr_multiplier_for_phase(cur_phase)
    for pg in optim.param_groups:
        pg["lr"] = base_lr * mult
    log.info(
        "phase transition: %d -> %d  (lr=%.2e, encoder_frozen=%s)",
        prev_phase, cur_phase, base_lr * mult, encoder_frozen,
    )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


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
) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 + (1 - SSIM-ish)."""
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
    return loss, parts


# ---------------------------------------------------------------------------
# BPTT detach helper
# ---------------------------------------------------------------------------


def detach_field(field: GaussianField) -> GaussianField:
    """Return a clone of ``field`` with all SoA tensors detached.

    BPTT-detach contract: training graph length = 1 frame; the only gradient
    path bridging t and t+1 is via ``temporal_consistency_loss(out_t+1, out_t)``.
    """
    out = field.clone()
    out.mu = out.mu.detach()
    out.log_scale = out.log_scale.detach()
    out.rotation = out.rotation.detach()
    out.color = out.color.detach()
    out.opacity = out.opacity.detach()
    # Detach every snapshot in history too — they may carry grad fns from
    # earlier steps.
    new_history = []
    for older in field.history:
        det = older.clone()
        det.mu = det.mu.detach()
        det.log_scale = det.log_scale.detach()
        det.rotation = det.rotation.detach()
        det.color = det.color.detach()
        det.opacity = det.opacity.detach()
        new_history.append(det)
    out._history.clear()
    # Push in reverse so the iteration order matches deque newest-first.
    for older in reversed(new_history):
        out._history.appendleft(older)
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_datasets(args: argparse.Namespace):
    """Build TrajectoryWindowDataset loaders for TartanAir + Sintel.

    Returns ``(tartan_loader_or_None, sintel_loader_or_None)``. Either may be
    None when its root flag wasn't provided. Phase-aware loader-selection is
    done per step in the main loop.
    """
    from torch.utils.data import DataLoader

    from oss.gaussian.data import (
        SintelGaussianDataset,
        TartanAirGaussianDataset,
    )
    from oss.sr.gaussian_temporal import (
        TrajectoryWindowDataset,
        default_collate_window,
    )
    from oss.sr.temporal import adapt_sintel, adapt_tartanair

    tartan_loader = None
    sintel_loader = None

    if args.tartanair_root is not None:
        ds_t = TartanAirGaussianDataset(root=args.tartanair_root, scale=2.0)
        ds_t = adapt_tartanair(ds_t)
        win_t = TrajectoryWindowDataset(ds_t, window=args.window)
        tartan_loader = DataLoader(
            win_t, batch_size=1, shuffle=True,
            num_workers=2, collate_fn=default_collate_window, drop_last=True,
        )
    if args.sintel_root is not None:
        ds_s = SintelGaussianDataset(root=args.sintel_root, scale=2.0, pass_name="clean")
        ds_s = adapt_sintel(ds_s)
        win_s = TrajectoryWindowDataset(ds_s, window=args.window)
        sintel_loader = DataLoader(
            win_s, batch_size=1, shuffle=True,
            num_workers=2, collate_fn=default_collate_window, drop_last=True,
        )
    return tartan_loader, sintel_loader


def select_loader_for_phase(phase: int, tartan_loader, sintel_loader):
    """Pick the right loader given the current phase.

    - Phase 1 / 2: TartanAir preferred (broad coverage), fallback Sintel.
    - Phase 3:     mix (alternated step-parity in main loop), but this fn
                   returns the default. Caller alternates explicitly.
    - Phase 4:     Sintel-only fine-tune.
    """
    if phase == 4:
        return sintel_loader or tartan_loader
    return tartan_loader or sintel_loader


# ---------------------------------------------------------------------------
# Synthetic batch (smoke mode)
# ---------------------------------------------------------------------------


def synthetic_window_batch(
    window: int, height: int, width: int, scale: int, device: str,
) -> dict[str, Any]:
    """Synthetic multi-frame moving-rectangle window batch.

    Returns a structure matching ``default_collate_window`` output with
    ``B=1``. Each frame is dict-of-tensors. Motion is a constant +1 LR-pixel
    shift in x — easy to reason about for the temporal-consistency check.
    """
    H_hr, W_hr = height * scale, width * scale
    g = torch.Generator(device=device).manual_seed(int(time.time()) & 0xFFFF)

    frames = []
    for k in range(window):
        # Rectangle moves by k pixels in x.
        lr = torch.zeros(1, 3, height, width, device=device)
        x0 = 4 + k
        x1 = min(width, x0 + 8)
        if x1 > x0:
            lr[:, :, 4:12, x0:x1] = 0.7
        gt_hr = F.interpolate(lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
        depth = torch.empty(1, 1, height, width, device=device).uniform_(0, 1, generator=g)
        motion = torch.zeros(1, 2, height, width, device=device)
        motion[:, 0] = 1.0  # forward flow t -> t+1: +1 LR-pixel in x
        normals = torch.empty(1, 3, height, width, device=device).uniform_(-1, 1, generator=g)
        canvas_hint = torch.zeros(1, 3, height, width, device=device)
        frames.append({
            "lr_frame": lr,
            "depth": depth,
            "motion": motion,
            "normals": normals,
            "canvas_hint": canvas_hint,
            "gt_hr_frame": gt_hr,
        })
    return {"frames": frames, "trajectory_key": ["smoke"]}


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def _make_12ch_input(
    lr: torch.Tensor, depth: torch.Tensor, motion: torch.Tensor,
    normals: torch.Tensor, canvas: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the 12-channel network input the encoder expects."""
    return torch.cat([lr, depth, motion, normals, canvas], dim=1)


def _frame_to_inputs(
    frame: dict[str, torch.Tensor], device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull (x_12ch, motion_lr, gt_hr) out of a collated frame dict.

    Frame fields follow the gaussian dataset convention:
        ``lr_frame``, ``depth``, ``motion``, ``normals``, ``canvas_hint``,
        ``gt_hr_frame``.
    """
    lr = frame["lr_frame"].to(device)
    depth = frame["depth"].to(device)
    motion = frame["motion"].to(device)
    normals = frame["normals"].to(device)
    canvas = frame["canvas_hint"].to(device)
    gt = frame["gt_hr_frame"].to(device)
    x12 = _make_12ch_input(lr, depth, motion, normals, canvas)
    return x12, motion, gt


def train_step(
    model: GaussianTemporalSRModel,
    batch: dict[str, Any],
    optim: torch.optim.Optimizer,
    *,
    phase: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """One optimizer step over an N-frame trajectory window.

    Walks the window frame-by-frame:
      - Phase 1: ``prev_field`` is always None (single-frame fitter).
      - Phase 2+: carry detached ``prev_field`` across frames.
      - Phase 3+: temporal-consistency + regularization loss active.

    Motion convention: for the t -> t+1 transition we feed the flow stored
    AT frame t (forward flow t -> t+1). Consistent with the pixel-fix.
    """
    device = args.device
    frames = batch["frames"]
    n_frames = len(frames)
    if n_frames < 1:
        raise ValueError("empty window")

    optim.zero_grad()
    total_loss: torch.Tensor = torch.zeros((), device=device)
    parts: dict[str, float] = {}

    prev_field: Optional[GaussianField] = None
    out_prev: Optional[torch.Tensor] = None
    field_prev_for_reg: Optional[GaussianField] = None
    flow_prev: Optional[torch.Tensor] = None  # flow t -> t+1 stored AT frame t
    n_app_terms = 0

    for idx in range(n_frames):
        x12, motion_t, gt = _frame_to_inputs(frames[idx], device=device)

        # Motion fed to the model on the (idx-1)->idx transition is the flow
        # stored at frame (idx-1). Phase 1 ignores prev_field anyway.
        motion_for_model = (
            flow_prev if (phase >= 2 and flow_prev is not None) else motion_t
        )

        feed_prev = prev_field if phase >= 2 else None
        out_hr, new_field, _dbg = model(
            lr_inputs=x12, motion_lr=motion_for_model, prev_field=feed_prev,
        )

        # Appearance loss on every frame.
        loss_app, parts_app = appearance_loss(
            out_hr, gt, w_l1=1.0, w_ssim=args.ssim_weight,
        )
        total_loss = total_loss + loss_app
        n_app_terms += 1
        for k, v in parts_app.items():
            parts[f"f{idx}_{k}"] = v

        # Temporal consistency between consecutive renders (Phase 3+).
        # The flow used is the flow stored at the previous frame
        # (forward flow (idx-1) -> idx) — same convention as the pixel fix.
        if phase >= 3 and out_prev is not None and flow_prev is not None:
            tc = temporal_consistency_loss(
                out_hr, out_prev, flow_prev, scale_factor=float(model.scale),
            )
            total_loss = total_loss + args.tc_weight * tc
            parts[f"f{idx}_tc"] = float(tc.item())

        # Regularization (Phase 3+).
        if phase >= 3 and field_prev_for_reg is not None:
            reg = gaussian_regularization_loss(
                field_t=new_field,
                field_t_minus_1=field_prev_for_reg,
                max_area=args.max_area,
                max_count=args.max_count,
            )
            total_loss = total_loss + args.reg_weight * reg
            parts[f"f{idx}_reg"] = float(reg.item())

        # BPTT-detach: store detached field as prev for next step. The only
        # cross-frame gradient path is via temporal_consistency_loss, which
        # consumes ``out_prev`` (a tensor, NOT detached for the duration of
        # this single training step but recomputed on each step).
        out_prev = out_hr
        prev_field = detach_field(new_field) if phase >= 2 else None
        field_prev_for_reg = detach_field(new_field) if phase >= 3 else None
        flow_prev = motion_t.detach()

    if n_app_terms > 0:
        total_loss = total_loss / float(n_app_terms)

    if not torch.isfinite(total_loss):
        # Surface the bad parts to the caller. Don't backward.
        parts["loss"] = float("nan")
        parts["phase"] = float(phase)
        return parts

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
    )
    optim.step()

    parts["loss"] = float(total_loss.item())
    parts["phase"] = float(phase)
    return parts


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirror sr_train_temporal.py)
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: GaussianTemporalSRModel,
    optim: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    """Save a rolling training checkpoint."""
    ckpt_path = output_dir / f"step-{step:08d}.pt"
    payload: dict[str, Any] = {
        "step": step,
        "kind": "gaussian_temporal",
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "gaussian_temporal_model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(payload, ckpt_path)
    log.info("ckpt -> %s", ckpt_path)
    return ckpt_path


def load_latest_checkpoint(
    output_dir: Path,
    model: GaussianTemporalSRModel,
    optim: torch.optim.Optimizer,
    device: str,
) -> tuple[int, list[dict], list[dict]]:
    """Auto-resume from the most recent ``step-*.pt`` in ``output_dir``."""
    if not output_dir.exists():
        return 0, [], []
    ckpts = sorted(output_dir.glob("step-*.pt"))
    if not ckpts:
        return 0, [], []
    latest = ckpts[-1]
    log.info("auto-resume: loading %s", latest)
    ck = torch.load(latest, map_location=device, weights_only=False)
    if "gaussian_temporal_model" in ck:
        model.load_state_dict(ck["gaussian_temporal_model"])
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
    log.info(
        "resumed at step=%d (metrics=%d, score=%d)",
        resume_step, len(metrics_log), len(score_log),
    )
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


def _approx_psnr_from_l1(l1) -> float:
    """Cheap PSNR proxy from L1 (NOT a real metric). Just for curve shape on
    the dashboard until ``scripts/sr_gaussian_temporal_held_out.py`` writes
    real PSNR.
    """
    if l1 is None or l1 <= 0:
        return float("nan")
    return float(-20.0 * math.log10(max(float(l1), 1e-6)))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def build_optimizer(model: GaussianTemporalSRModel, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


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
        "v5-gaussian-temporal: device=%s steps=%d window=%d smoke=%s "
        "phase1_end=%d phase2_end=%d phase3_end=%d lr=%.2e",
        args.device, args.max_steps, args.window, args.smoke,
        args.phase1_end, args.phase2_end, args.phase3_end, args.lr,
    )

    # Build model.
    model = GaussianTemporalSRModel(
        in_channels=12, scale=2, max_count=args.max_count,
    )
    model.to(args.device)
    log.info("model params: total=%d", sum(p.numel() for p in model.parameters()))

    optim = build_optimizer(model, lr=args.lr)

    # Auto-resume.
    resume_step, metrics_log, score_log = load_latest_checkpoint(
        args.output_dir, model, optim, args.device,
    )

    # Initial phase application.
    cur_phase = phase_for_step(
        max(resume_step, 0), args.phase1_end, args.phase2_end, args.phase3_end,
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
            log.error(
                "Non-smoke training requires --tartanair-root and/or --sintel-root.",
            )
            return 2

    step = resume_step
    parts: dict[str, float] = {}
    while step < args.max_steps:
        step += 1
        final_step = step

        new_phase = phase_for_step(
            step, args.phase1_end, args.phase2_end, args.phase3_end,
        )
        if new_phase != cur_phase:
            apply_phase(
                model, optim, args.lr,
                prev_phase=cur_phase, cur_phase=new_phase,
            )
            cur_phase = new_phase

        # Pull a batch.
        if args.smoke:
            batch = synthetic_window_batch(
                window=args.window, height=16, width=16,
                scale=model.scale, device=args.device,
            )
        else:
            loader = select_loader_for_phase(cur_phase, tartan_loader, sintel_loader)
            if loader is None:
                log.error("No loader available for phase %d", cur_phase)
                return 3
            # Phase 3: alternate TartanAir/Sintel by step parity if both exist.
            if (cur_phase == 3 and tartan_loader is not None
                    and sintel_loader is not None):
                use_sintel = (step % 2 == 0)
                if use_sintel:
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
                "step=%d phase=%d loss=%.4f f0_l1=%.4f",
                step, cur_phase, row["loss"],
                row.get("f0_l1", float("nan")),
            )

        # Periodic checkpoint + rolling metrics dump.
        # Dashboard schema (scripts/training_dashboard.py): score_log entries
        # must expose ``model_psnr_mean``, ``bicubic_psnr_mean``,
        # ``model_lpips_mean``, ``bicubic_lpips_mean``. Bicubic fields are
        # None at training time; the held-out script writes real values.
        if step % args.ckpt_every == 0 or step == args.max_steps or args.smoke:
            save_checkpoint(args.output_dir, step, model, optim, args)
            score_log.append({
                "step": step,
                "loss": float(parts["loss"]),
                "phase": int(cur_phase),
                "model_psnr_mean": _approx_psnr_from_l1(
                    parts.get("f0_l1", parts["loss"])
                ),
                "bicubic_psnr_mean": None,
                "model_lpips_mean": None,
                "bicubic_lpips_mean": None,
            })
            dump_metrics(args.output_dir, metrics_log, score_log)

    # Final dump (idempotent).
    if final_step > 0:
        save_checkpoint(args.output_dir, final_step, model, optim, args)
        if not score_log or score_log[-1]["step"] != final_step:
            score_log.append({
                "step": final_step,
                "loss": float(parts.get("loss", float("nan"))),
                "phase": int(cur_phase),
                "model_psnr_mean": _approx_psnr_from_l1(
                    parts.get("f0_l1", parts.get("loss", 1.0))
                ),
                "bicubic_psnr_mean": None,
                "model_lpips_mean": None,
                "bicubic_lpips_mean": None,
            })
        dump_metrics(args.output_dir, metrics_log, score_log)

    elapsed = time.monotonic() - train_start
    final_loss = parts.get("loss", float("nan"))
    # WMI-orphan-spawn-friendly: flush after every print so downstream readers
    # see the script's progress in real time.
    print(f"v5-gaussian-temporal training: device={args.device} smoke={args.smoke}", flush=True)
    print(f"final_step={final_step} elapsed={elapsed:.1f}s", flush=True)
    print(f"final_loss={final_loss:.6f} phase={cur_phase}", flush=True)
    print(f"checkpoint -> {args.output_dir}/step-{final_step:08d}.pt", flush=True)
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""OSS-Gaussian param network training entrypoint.

Sprint 4 / T4.5. Reads training data via the dataset adapters in
`oss/gaussian/data/`, feeds the encoder/decoder network in
`oss/gaussian/network/`, decodes via OutputHead -> GaussianBatch, renders
via the Sprint 1 Rasterizer, computes composite loss, backprops through
the differentiable renderer.

Usage:
    python -m oss.gaussian.train.train --tier standard --max-steps 100000 \\
        --output-dir checkpoints/gaussian-standard-001 \\
        --dataset-root ~/datasets

    # Smoke test (gates Lambda H100 spend per 2026-05-01 validation memo):
    python -m oss.gaussian.train.train --smoke-test \\
        --sintel-sequence alley_1 \\
        --dataset-root ~/datasets \\
        --output-dir checkpoints/sprint4-smoke

    # CI sanity (no real data required):
    python -m oss.gaussian.train.train --use-synthetic-batch \\
        --max-steps 5 --output-dir /tmp/oss-ci-sanity

Honest current scope: this is the v0 training loop. It exercises the
end-to-end pipeline on synthetic fixture data and small real subsets.
The full Sprint 4 ablation (bank size, K per tile, tier transfer) and
the production-quality multi-day training runs are subsequent work.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from oss.gaussian.network import (
    CovariancePriorBank,
    GaussianParamNetwork,
    OutputHead,
)
from oss.gaussian.network.param_net import TIER_CONFIGS, param_net_for_tier
from oss.gaussian.renderer import Rasterizer

log = logging.getLogger("oss.gaussian.train")


# ---------------------------------------------------------------------------
# TrainArgs -- all configuration in one frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainArgs:
    tier: str
    max_steps: int
    batch_size: int
    learning_rate: float
    output_dir: Path
    dataset_root: Path
    bank_size: int
    k_per_tile: int
    log_every: int
    ckpt_every: int
    seed: int
    device: str
    # Real-data flags
    use_synthetic_batch: bool
    dataset: str             # "sintel" | "srgd"
    sintel_sequence: Optional[str]
    srgd_scene: Optional[str]
    force_lr_synth: bool
    enable_gbuffer_bias: bool
    enable_engine_aliased_lr: bool
    score_every: int         # run bicubic-vs-model comparison every N steps
    # Time-bounding
    max_time_seconds: Optional[int]
    # Smoke-test mode (implies pico tier, batch=2, 3-hr kill, bicubic comparison, real data)
    smoke_test: bool

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "TrainArgs":
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--tier", choices=list(TIER_CONFIGS), default="standard")
        p.add_argument("--max-steps", type=int, default=10_000)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--dataset-root", type=Path, default=Path.home() / "datasets")
        p.add_argument("--bank-size", type=int, default=16)
        p.add_argument("--k-per-tile", type=int, default=5)
        p.add_argument("--log-every", type=int, default=20)
        p.add_argument("--ckpt-every", type=int, default=2_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        # Real-data flags
        p.add_argument(
            "--use-synthetic-batch",
            action="store_true",
            default=False,
            help="Use random synthetic tensors instead of real Sintel data (CI sanity path).",
        )
        p.add_argument(
            "--dataset",
            choices=["sintel", "srgd"],
            default="sintel",
            help="Real-data dataset adapter to use (default: sintel).",
        )
        p.add_argument(
            "--sintel-sequence",
            type=str,
            default=None,
            help=(
                "Restrict training to a single Sintel sequence name (e.g. alley_1). "
                "Used only when --dataset=sintel."
            ),
        )
        p.add_argument(
            "--srgd-scene",
            type=str,
            default=None,
            help=(
                "Restrict training to a single SRGD scene name (e.g. ActionRPG). "
                "Used only when --dataset=srgd."
            ),
        )
        p.add_argument(
            "--force-lr-synth",
            action="store_true",
            default=False,
            help=(
                "Ignore any pre-baked LR files on disk and always synthesize "
                "LR from HR via lr_synth. Avoids the bicubic-LR-trap from "
                "datasets that ship bicubic-downsampled LR."
            ),
        )
        p.add_argument(
            "--enable-gbuffer-bias",
            action="store_true",
            default=False,
            help="Pass depth+normals G-buffers into OutputHead (anisotropic covariance).",
        )
        p.add_argument(
            "--enable-engine-aliased-lr",
            action="store_true",
            default=False,
            help="Wrap dataset with EngineAliasedLRSynth (jitter+TAA blur) for engine-realistic LR.",
        )
        p.add_argument(
            "--eval-every",
            type=int,
            default=500,
            dest="score_every",
            help="Run bicubic-vs-model PSNR comparison every N steps (0 = disabled).",
        )
        p.add_argument(
            "--max-time-seconds",
            type=int,
            default=None,
            help="Wall-clock kill switch: stop training after this many seconds.",
        )
        p.add_argument(
            "--smoke-test",
            action="store_true",
            default=False,
            help=(
                "Low-capacity smoke test mode: pico tier, batch=2, 3-hr kill, "
                "bicubic comparison enabled, real Sintel data. "
                "Gates Lambda H100 spend per 2026-05-01 validation memo Decision 1."
            ),
        )
        a = p.parse_args(argv)

        # Smoke-test overrides applied after arg parsing.
        smoke_test = a.smoke_test
        tier = a.tier
        batch_size = a.batch_size
        max_time_seconds = a.max_time_seconds
        enable_gbuffer_bias = a.enable_gbuffer_bias
        enable_engine_aliased_lr = a.enable_engine_aliased_lr
        use_synthetic_batch = a.use_synthetic_batch
        score_every = a.score_every

        force_lr_synth = a.force_lr_synth
        if smoke_test:
            # Hard overrides: pico tier, small batch, 3-hour wall clock.
            tier = "pico"
            batch_size = 2
            if max_time_seconds is None:
                max_time_seconds = 10800  # 3 hours
            enable_gbuffer_bias = True
            enable_engine_aliased_lr = True
            force_lr_synth = True  # avoid bicubic-LR-trap on SRGD's pre-baked LR
            use_synthetic_batch = False  # smoke test requires real data

        return cls(
            tier=tier,
            max_steps=a.max_steps,
            batch_size=batch_size,
            learning_rate=a.learning_rate,
            output_dir=a.output_dir,
            dataset_root=a.dataset_root,
            bank_size=a.bank_size,
            k_per_tile=a.k_per_tile,
            log_every=a.log_every,
            ckpt_every=a.ckpt_every,
            seed=a.seed,
            device=a.device,
            use_synthetic_batch=use_synthetic_batch,
            dataset=a.dataset,
            sintel_sequence=a.sintel_sequence,
            srgd_scene=a.srgd_scene,
            force_lr_synth=force_lr_synth,
            enable_gbuffer_bias=enable_gbuffer_bias,
            enable_engine_aliased_lr=enable_engine_aliased_lr,
            score_every=score_every,
            max_time_seconds=max_time_seconds,
            smoke_test=smoke_test,
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def composite_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    w_l1: float = 1.0,
    w_ssim: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per Sprint 4 plan: HDR-aware L1 + (1 - SSIM). LPIPS deferred to a
    perceptual variant since LPIPS is heavy and slows the inner loop.

    Returns the scalar loss + a dict of components for logging.
    """
    l1 = F.l1_loss(rendered, target)
    rendered_lum = rendered.mean(dim=1, keepdim=True)
    target_lum = target.mean(dim=1, keepdim=True)
    mu_r = F.avg_pool2d(rendered_lum, 8, 8)
    mu_t = F.avg_pool2d(target_lum, 8, 8)
    ssim_proxy = 1.0 - F.l1_loss(mu_r, mu_t)
    loss = w_l1 * l1 + w_ssim * (1.0 - ssim_proxy)
    return loss, {"l1": float(l1.item()), "ssim_proxy": float(ssim_proxy.item())}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(args: TrainArgs) -> tuple[GaussianParamNetwork, OutputHead, CovariancePriorBank]:
    """Wire up Sprint 4 components.

    Note: per-tier K-per-tile is fixed by TIER_CONFIGS in param_net.py.
    """
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(args.tier, bank_size=args.bank_size)
    head = OutputHead(
        bank=bank,
        k_per_tile=net.k_per_tile,
        enable_gbuffer_bias=args.enable_gbuffer_bias,
    )
    return net, head, bank


# ---------------------------------------------------------------------------
# Synthetic batch (CI / sanity path)
# ---------------------------------------------------------------------------


def synthetic_batch(
    batch_size: int, height: int, width: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic input + GT pair for end-to-end sanity.

    Only used when --use-synthetic-batch is set. Real training uses
    build_dataloader() instead.
    """
    g = torch.Generator(device=device).manual_seed(int(time.time()) & 0xFFFF)
    x = torch.rand((batch_size, 12, height, width), generator=g, device=device)
    target = torch.rand((batch_size, 3, height * 2, width * 2), generator=g, device=device)
    return x, target


# ---------------------------------------------------------------------------
# Real DataLoader
# ---------------------------------------------------------------------------


def _build_lr_synth(args: TrainArgs):
    from oss.gaussian.data import EngineAliasedLRSynth
    if not args.enable_engine_aliased_lr:
        return None
    return EngineAliasedLRSynth(
        enable_jitter=True, enable_taa_blur=True, enable_jpeg=False
    )


def _build_sintel_dataset(args: TrainArgs):
    from oss.gaussian.data import SintelGaussianDataset

    candidate_roots = [
        args.dataset_root,
        args.dataset_root / "MPI-Sintel-complete",
        args.dataset_root / "sintel",
    ]
    sintel_root = None
    for cand in candidate_roots:
        if (cand / "training" / "clean").is_dir():
            sintel_root = cand
            break
    if sintel_root is None:
        raise FileNotFoundError(
            f"Sintel dataset not found. Looked under each of: "
            f"{[str(c) for c in candidate_roots]}. "
            f"Expected `<root>/training/clean/<sequence>/...` layout."
        )

    ds = SintelGaussianDataset(
        root=sintel_root,
        scale=2.0,
        pass_name="clean",
        lr_synth=_build_lr_synth(args),
    )
    if args.sintel_sequence:
        ds._items = [
            it for it in ds._items if it[0].parent.name == args.sintel_sequence
        ]
        if not ds._items:
            raise ValueError(
                f"No frames found for sequence {args.sintel_sequence!r} under "
                f"{sintel_root}. Check --dataset-root and --sintel-sequence."
            )
    return ds


def _build_srgd_dataset(args: TrainArgs):
    from oss.gaussian.data import SRGDGaussianDataset

    # Probe two layouts: a direct SRGD root or a `srgd` subdir.
    candidates = [args.dataset_root, args.dataset_root / "srgd"]
    srgd_root = None
    for cand in candidates:
        if (cand / "data" / "GameEngineData").is_dir() or (cand / "hr").is_dir():
            srgd_root = cand
            break
    if srgd_root is None:
        raise FileNotFoundError(
            f"SRGD dataset not found. Looked under: {[str(c) for c in candidates]}."
        )

    return SRGDGaussianDataset(
        root=srgd_root,
        scale=2.0,
        lr_synth=_build_lr_synth(args),
        scene=args.srgd_scene,
        force_synth_lr=args.force_lr_synth,
    )


def build_dataloader(args: TrainArgs):  # type: ignore[return]
    """Construct a DataLoader for the configured dataset (sintel | srgd)."""
    from oss.gaussian.data import collate_examples
    from torch.utils.data import DataLoader

    if args.dataset == "sintel":
        ds = _build_sintel_dataset(args)
    elif args.dataset == "srgd":
        ds = _build_srgd_dataset(args)
    else:
        raise ValueError(f"Unknown --dataset value: {args.dataset!r}")

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_examples,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Bicubic baseline comparison
# ---------------------------------------------------------------------------


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between two [0,1] tensors of any shape.

    MSE is clamped to >= 1e-12 to avoid inf on identical pairs.
    """
    mse = float(F.mse_loss(pred.float(), target.float()).item())
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


def _tile_align_batch(
    lr: torch.Tensor,
    depth: torch.Tensor,
    motion: torch.Tensor,
    normals: torch.Tensor,
    canvas: torch.Tensor,
    gt_hr: torch.Tensor,
    tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center-crop a batch so LR is a multiple of ``tile`` and HR aligns at the
    same scale ratio. Used for datasets whose native frame size doesn't divide
    cleanly (e.g. SRGD 540x960 → 270x480 LR).
    """
    scale_int = int(round(gt_hr.shape[-2] / lr.shape[-2]))
    lr_h, lr_w = lr.shape[-2:]
    lr_h_a = (lr_h // tile) * tile
    lr_w_a = (lr_w // tile) * tile
    if (lr_h_a, lr_w_a) == (lr_h, lr_w):
        return lr, depth, motion, normals, canvas, gt_hr
    top = (lr_h - lr_h_a) // 2
    left = (lr_w - lr_w_a) // 2
    lr = lr[..., top:top + lr_h_a, left:left + lr_w_a]
    depth = depth[..., top:top + lr_h_a, left:left + lr_w_a]
    motion = motion[..., top:top + lr_h_a, left:left + lr_w_a]
    normals = normals[..., top:top + lr_h_a, left:left + lr_w_a]
    canvas = canvas[..., top:top + lr_h_a, left:left + lr_w_a]
    hr_top = top * scale_int
    hr_left = left * scale_int
    gt_hr = gt_hr[
        ..., hr_top:hr_top + lr_h_a * scale_int, hr_left:hr_left + lr_w_a * scale_int
    ]
    return lr, depth, motion, normals, canvas, gt_hr


def evaluate_against_bicubic(
    net: GaussianParamNetwork,
    head: OutputHead,
    bank: CovariancePriorBank,
    dataloader,  # type: ignore[type-arg]
    device: str,
    n_samples: int = 8,
) -> dict:
    """Compare model output PSNR against bicubic upsample on held-out examples.

    Args:
        net:        Trained GaussianParamNetwork (set to .train() after call).
        head:       Wired OutputHead.
        bank:       CovariancePriorBank (unused directly; kept for API symmetry).
        dataloader: DataLoader yielding collated GaussianTrainingExample dicts.
        device:     torch device string.
        n_samples:  Maximum examples to score (may be fewer if dataset is smaller).

    Returns dict with keys:
        model_psnr_mean          (float)
        bicubic_psnr_mean        (float)
        model_psnr_per_sample    (list[float])
        bicubic_psnr_per_sample  (list[float])
        model_beats_bicubic_count (int)
    """
    renderer = Rasterizer()
    net.train(False)
    model_psnrs: list[float] = []
    bicubic_psnrs: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(model_psnrs) >= n_samples:
                break

            lr = batch["lr_frame"].to(device)
            depth = batch["depth"].to(device)
            motion = batch["motion"].to(device)
            normals = batch["normals"].to(device)
            canvas = batch["canvas_hint"].to(device)
            gt_hr = batch["gt_hr_frame"].to(device)

            lr, depth, motion, normals, canvas, gt_hr = _tile_align_batch(
                lr, depth, motion, normals, canvas, gt_hr, tile=net.tile_size
            )

            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            H_hr, W_hr = gt_hr.shape[-2:]

            # Bicubic baseline for the entire batch at once.
            bicubic_hr = F.interpolate(
                lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)

            raw = net(x)

            for b_idx in range(lr.shape[0]):
                if len(model_psnrs) >= n_samples:
                    break

                depth_b = depth[b_idx : b_idx + 1]
                normals_b = normals[b_idx : b_idx + 1]
                gaussians = head.to_gaussian_batch(
                    raw,
                    batch_index=b_idx,
                    depth=depth_b,
                    normals=normals_b,
                )
                rendered = renderer(gaussians, output_hw=(H_hr, W_hr)).clamp(0.0, 1.0)

                gt_single = gt_hr[b_idx]
                bicubic_single = bicubic_hr[b_idx]

                model_psnrs.append(_psnr(rendered, gt_single))
                bicubic_psnrs.append(_psnr(bicubic_single, gt_single))

    net.train(True)

    beats_count = sum(1 for m, b in zip(model_psnrs, bicubic_psnrs) if m > b)

    if model_psnrs:
        model_mean = float(sum(model_psnrs) / len(model_psnrs))
        bicubic_mean = float(sum(bicubic_psnrs) / len(bicubic_psnrs))
    else:
        model_mean = float("nan")
        bicubic_mean = float("nan")

    return {
        "model_psnr_mean": model_mean,
        "bicubic_psnr_mean": bicubic_mean,
        "model_psnr_per_sample": model_psnrs,
        "bicubic_psnr_per_sample": bicubic_psnrs,
        "model_beats_bicubic_count": beats_count,
    }


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    output_dir: Path,
    step: int,
    tier: str,
    net: GaussianParamNetwork,
    bank: CovariancePriorBank,
    args: TrainArgs,
) -> None:
    ckpt_path = output_dir / f"step-{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "tier": tier,
            "net": net.state_dict(),
            "bank": bank.state_dict(),
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in args.__dict__.items()
            },
        },
        ckpt_path,
    )
    log.info("ckpt -> %s", ckpt_path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = TrainArgs.from_cli(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    log.info(
        "device=%s tier=%s steps=%d batch=%d smoke=%s synth=%s",
        args.device,
        args.tier,
        args.max_steps,
        args.batch_size,
        args.smoke_test,
        args.use_synthetic_batch,
    )
    if args.max_time_seconds:
        log.info(
            "wall-clock kill: %d s (%.1f hr)",
            args.max_time_seconds,
            args.max_time_seconds / 3600.0,
        )

    net, head, bank = build_model(args)
    net.to(args.device)
    bank.to(args.device)
    if args.enable_gbuffer_bias and head.gbuffer_bias is not None:
        head.gbuffer_bias.to(args.device)

    renderer = Rasterizer()
    optim = torch.optim.AdamW(
        list(net.parameters()) + list(bank.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    log.info(
        "net params=%d bank params=%d",
        sum(p.numel() for p in net.parameters()),
        sum(p.numel() for p in bank.parameters()),
    )

    metrics_log: list[dict] = []
    score_log: list[dict] = []
    train_start = time.monotonic()
    timed_out = False

    # ------------------------------------------------------------------
    # Synthetic-batch path (CI / sanity -- no real data needed)
    # ------------------------------------------------------------------
    if args.use_synthetic_batch:
        h, w = 64, 64
        log.info("using synthetic_batch path (no real data)")
        final_step = 0
        for step in range(1, args.max_steps + 1):
            if args.max_time_seconds is not None:
                elapsed = time.monotonic() - train_start
                if elapsed > args.max_time_seconds:
                    log.info("wall-clock limit at step %d (%.1f s)", step, elapsed)
                    timed_out = True
                    step -= 1
                    break

            x, target = synthetic_batch(args.batch_size, h, w, args.device)
            optim.zero_grad()

            raw = net(x)
            rendered_batch = []
            for b_idx in range(x.shape[0]):
                gaussians = head.to_gaussian_batch(raw, batch_index=b_idx)
                rendered_batch.append(renderer(gaussians, output_hw=(h * 2, w * 2)))
            rendered = torch.stack(rendered_batch, dim=0)

            loss, parts = composite_loss(rendered, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(bank.parameters()), max_norm=1.0
            )
            optim.step()

            if step % args.log_every == 0:
                row = {"step": step, "loss": float(loss.item()), **parts}
                metrics_log.append(row)
                log.info(
                    "step=%d loss=%.4f l1=%.4f ssim_proxy=%.4f",
                    step,
                    row["loss"],
                    row["l1"],
                    row["ssim_proxy"],
                )

            if step % args.ckpt_every == 0 or step == args.max_steps:
                _save_checkpoint(args.output_dir, step, args.tier, net, bank, args)

            final_step = step

    # ------------------------------------------------------------------
    # Real-data path (Sintel + EngineAliasedLRSynth)
    # ------------------------------------------------------------------
    else:
        loader = build_dataloader(args)
        score_loader = build_dataloader(args)
        log.info(
            "dataset size=%d sequence_filter=%r",
            len(loader.dataset),  # type: ignore[arg-type]
            args.sintel_sequence,
        )

        step = 0
        final_step = 0
        data_iter = iter(loader)

        while step < args.max_steps:
            # Wall-clock kill switch.
            if args.max_time_seconds is not None:
                elapsed = time.monotonic() - train_start
                if elapsed > args.max_time_seconds:
                    log.info("wall-clock limit at step %d (%.1f s)", step, elapsed)
                    timed_out = True
                    break

            # Cycle the DataLoader when exhausted.
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            step += 1
            final_step = step

            lr = batch["lr_frame"].to(args.device)
            depth = batch["depth"].to(args.device)
            motion = batch["motion"].to(args.device)
            normals = batch["normals"].to(args.device)
            canvas = batch["canvas_hint"].to(args.device)
            gt_hr = batch["gt_hr_frame"].to(args.device)

            lr, depth, motion, normals, canvas, gt_hr = _tile_align_batch(
                lr, depth, motion, normals, canvas, gt_hr, tile=net.tile_size
            )

            # 12-channel input: LR(3)+depth(1)+motion(2)+normals(3)+canvas(3).
            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            H_hr, W_hr = gt_hr.shape[-2:]

            optim.zero_grad()
            raw = net(x)

            rendered_batch = []
            for b_idx in range(lr.shape[0]):
                depth_b = depth[b_idx : b_idx + 1]
                normals_b = normals[b_idx : b_idx + 1]
                gaussians = head.to_gaussian_batch(
                    raw,
                    batch_index=b_idx,
                    depth=depth_b,
                    normals=normals_b,
                )
                rendered_batch.append(renderer(gaussians, output_hw=(H_hr, W_hr)))
            rendered = torch.stack(rendered_batch, dim=0)

            loss, parts = composite_loss(rendered, gt_hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(bank.parameters()), max_norm=1.0
            )
            optim.step()

            if step % args.log_every == 0:
                row = {"step": step, "loss": float(loss.item()), **parts}
                metrics_log.append(row)
                log.info(
                    "step=%d loss=%.4f l1=%.4f ssim_proxy=%.4f",
                    step,
                    row["loss"],
                    row["l1"],
                    row["ssim_proxy"],
                )

            if step % args.ckpt_every == 0:
                _save_checkpoint(args.output_dir, step, args.tier, net, bank, args)

            # Periodic bicubic comparison.
            if args.score_every > 0 and step % args.score_every == 0:
                log.info("--- bicubic comparison at step %d ---", step)
                result = evaluate_against_bicubic(
                    net, head, bank, score_loader, args.device, n_samples=8
                )
                score_row = {"step": step, **result}
                score_log.append(score_row)
                log.info(
                    "step=%d model_psnr=%.2f dB  bicubic_psnr=%.2f dB  "
                    "beats_bicubic=%d/8",
                    step,
                    result["model_psnr_mean"],
                    result["bicubic_psnr_mean"],
                    result["model_beats_bicubic_count"],
                )

        # Final checkpoint (covers both natural end and timeout).
        _save_checkpoint(args.output_dir, final_step, args.tier, net, bank, args)

        # Final bicubic comparison (always at end of real-data training).
        log.info("--- final bicubic comparison at step %d ---", final_step)
        final_result = evaluate_against_bicubic(
            net, head, bank, score_loader, args.device, n_samples=8
        )
        score_log.append({"step": final_step, "final": True, **final_result})
        log.info(
            "FINAL model_psnr=%.2f dB  bicubic_psnr=%.2f dB  beats_bicubic=%d/8",
            final_result["model_psnr_mean"],
            final_result["bicubic_psnr_mean"],
            final_result["model_beats_bicubic_count"],
        )

        # Smoke-test gate per 2026-05-01 validation memo Decision 1.
        if args.smoke_test:
            passed = final_result["model_beats_bicubic_count"] > 0
            verdict = "PASS" if passed else "FAIL"
            log.info("SMOKE TEST RESULT: %s", verdict)
            print(
                f"\nSMOKE TEST RESULT: {verdict}\n"
                f"  model_psnr  = {final_result['model_psnr_mean']:.2f} dB\n"
                f"  bicubic_psnr= {final_result['bicubic_psnr_mean']:.2f} dB\n"
                f"  beats_bicubic = {final_result['model_beats_bicubic_count']}/8"
            )

    # ------------------------------------------------------------------
    # Write metrics to disk.
    # ------------------------------------------------------------------
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"train": metrics_log, "score": score_log}, f, indent=2)
    log.info("metrics -> %s", metrics_path)

    elapsed_total = time.monotonic() - train_start
    log.info("done: steps=%d elapsed=%.1f s", final_step, elapsed_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

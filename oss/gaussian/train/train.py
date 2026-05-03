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
    PixelResidualHead,
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
    renderer_backend: str   # "auto" | "cuda" | "reference"
    enable_gbuffer_bias: bool
    enable_engine_aliased_lr: bool
    lr_synth_blur_sigma: float
    lr_synth_jpeg: bool
    lr_synth_jpeg_quality: int
    lpips_loss_weight: float
    enable_pixel_residual: bool
    pixel_residual_hidden: int
    score_every: int         # run bicubic-vs-model comparison every N steps
    # Time-bounding
    max_time_seconds: Optional[int]
    # Smoke-test mode (implies pico tier, batch=2, 3-hr kill, bicubic comparison, real data)
    smoke_test: bool
    # SR-track fields (no effect when model_kind == "gaussian")
    model_kind: str   # "gaussian" | "sr_cnn" | "sr_rrdb"
    sr_backbone: str  # "simple" | "rrdb" (used only when model_kind == "sr_cnn"/"sr_rrdb")

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
            "--renderer-backend",
            choices=["auto", "cuda", "reference"],
            default="auto",
            help=(
                "Force the rasterizer backend. 'reference' (pure PyTorch) is "
                "slower but has verified backward; 'cuda' (gsplat 1.4.0) is "
                "much faster but its backward path has known issues."
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
            "--lr-synth-blur-sigma",
            type=float,
            default=0.5,
            help=(
                "TAA blur kernel sigma when --enable-engine-aliased-lr is on. "
                "0.5 = mild (≈DLSS), 1.0–1.5 = aggressive (drops the bicubic "
                "baseline and gives the SR network more to learn)."
            ),
        )
        p.add_argument(
            "--lr-synth-jpeg",
            action="store_true",
            default=False,
            help="Apply JPEG artifact pass in engine-aliased LR synth.",
        )
        p.add_argument(
            "--lr-synth-jpeg-quality",
            type=int,
            default=85,
            help="JPEG quality for --lr-synth-jpeg (1-95).",
        )
        p.add_argument(
            "--lpips-loss-weight",
            type=float,
            default=0.0,
            help=(
                "When > 0, add w_lpips * LPIPS-VGG to the composite loss "
                "(Real-ESRGAN-style perceptual training). Typical: 0.1. "
                "Costs ~1.5-2x training speed but pushes perceptual quality "
                "(LPIPS metric) significantly. Default 0 = disabled."
            ),
        )
        p.add_argument(
            "--enable-pixel-residual",
            action="store_true",
            default=False,
            help=(
                "V0.5: add a small CNN that predicts a per-pixel RGB "
                "residual on top of the splat-rendered output. Required to "
                "escape the V0 plateau per "
                "docs/superpowers/experiments/2026-05-02-output-head-dead-init.md."
            ),
        )
        p.add_argument(
            "--pixel-residual-hidden",
            type=int,
            default=32,
            help="Hidden channel count for the pixel-residual CNN (default 32).",
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
        # ---------------------------------------------------------------------------
        # SR-track model selection (2026-05-02 pivot, post-sprint-4 falsification)
        # ---------------------------------------------------------------------------
        p.add_argument(
            "--model",
            choices=["gaussian", "sr_cnn", "sr_rrdb"],
            default="gaussian",
            dest="model_kind",
            help=(
                "Training target model. 'gaussian' (default) trains the Gaussian-splat "
                "param network (OSS-RR track). 'sr_cnn' trains SRCNNSimple (OSS-SR "
                "CNN track). 'sr_rrdb' trains SRRRDB (OSS-SR RRDB variant). "
                "The gaussian path is bit-identical to the pre-2026-05-02 behavior."
            ),
        )
        p.add_argument(
            "--sr-backbone",
            choices=["simple", "rrdb"],
            default="simple",
            dest="sr_backbone",
            help=(
                "SR backbone variant.  Only used when --model is sr_cnn or sr_rrdb. "
                "'simple' uses SRCNNSimple; 'rrdb' uses SRRRDB."
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
        lr_synth_blur_sigma = a.lr_synth_blur_sigma
        lr_synth_jpeg = a.lr_synth_jpeg
        lr_synth_jpeg_quality = a.lr_synth_jpeg_quality
        enable_pixel_residual = a.enable_pixel_residual
        pixel_residual_hidden = a.pixel_residual_hidden
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
            # Aggressive engine-aliased synth: drop the bicubic baseline so the
            # SR network has something meaningful to learn against.
            if a.lr_synth_blur_sigma == 0.5:  # default → upgrade for smoke
                lr_synth_blur_sigma = 1.5
            lr_synth_jpeg = True if not a.lr_synth_jpeg else a.lr_synth_jpeg

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
            renderer_backend=a.renderer_backend,
            enable_gbuffer_bias=enable_gbuffer_bias,
            enable_engine_aliased_lr=enable_engine_aliased_lr,
            lr_synth_blur_sigma=lr_synth_blur_sigma,
            lr_synth_jpeg=lr_synth_jpeg,
            lr_synth_jpeg_quality=lr_synth_jpeg_quality,
            lpips_loss_weight=a.lpips_loss_weight,
            enable_pixel_residual=enable_pixel_residual,
            pixel_residual_hidden=a.pixel_residual_hidden,
            score_every=score_every,
            max_time_seconds=max_time_seconds,
            smoke_test=smoke_test,
            model_kind=a.model_kind,
            sr_backbone=a.sr_backbone,
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


_SSIM_FN = None
_SSIM_IMPORT_TRIED = False


def _get_ssim_fn():
    """Lazy-load pytorch_msssim. Returns None when missing (e.g. on CI)."""
    global _SSIM_FN, _SSIM_IMPORT_TRIED
    if _SSIM_IMPORT_TRIED:
        return _SSIM_FN
    _SSIM_IMPORT_TRIED = True
    try:
        from pytorch_msssim import ssim  # type: ignore[import-not-found]
        _SSIM_FN = ssim
    except ImportError:
        _SSIM_FN = None
    return _SSIM_FN


def composite_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    w_l1: float = 1.0,
    w_ssim: float = 0.1,
    w_lpips: float = 0.0,
    lpips_device: str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 + (1 - SSIM) [+ LPIPS-VGG] composite loss.

    When ``w_lpips > 0`` and the lpips package is importable, adds a
    LPIPS-VGG perceptual loss term (Real-ESRGAN-style). LPIPS-VGG runs
    a VGG-16 forward+backward on each step, so per-step compute is
    ~1.5-2x slower than L1+SSIM alone — but the perceptual quality
    lift is meaningful (drops LPIPS from ~0.39 to ~0.25 in published
    benchmarks).

    Returns the scalar loss + a dict of components for logging.
    """
    l1 = F.l1_loss(rendered, target)

    ssim_fn = _get_ssim_fn()
    if ssim_fn is not None:
        # pytorch_msssim.ssim expects values in [0, 1], shape (B, C, H, W).
        ssim_val = ssim_fn(
            rendered.clamp(0.0, 1.0),
            target.clamp(0.0, 1.0),
            data_range=1.0,
            size_average=True,
        )
        loss = w_l1 * l1 + w_ssim * (1.0 - ssim_val)
        parts: dict[str, float] = {"l1": float(l1.item()), "ssim": float(ssim_val.item())}
    else:
        # Fallback: pooled-L1 of luminance — mathematically distinct from
        # SSIM, but cheap and dependency-free for CI.
        rendered_lum = rendered.mean(dim=1, keepdim=True)
        target_lum = target.mean(dim=1, keepdim=True)
        mu_r = F.avg_pool2d(rendered_lum, 8, 8)
        mu_t = F.avg_pool2d(target_lum, 8, 8)
        pooled_l1 = F.l1_loss(mu_r, mu_t)
        loss = w_l1 * l1 + w_ssim * pooled_l1
        parts = {"l1": float(l1.item()), "pooled_l1": float(pooled_l1.item())}

    if w_lpips > 0:
        device = lpips_device or str(rendered.device)
        lpips_fn = _get_lpips_fn(device)
        if lpips_fn is not None:
            # LPIPS expects inputs scaled to [-1, 1].
            p = (rendered.clamp(0.0, 1.0) * 2 - 1)
            t = (target.clamp(0.0, 1.0) * 2 - 1)
            lpips_val = lpips_fn(p, t).mean()
            loss = loss + w_lpips * lpips_val
            parts["lpips"] = float(lpips_val.item())

    return loss, parts


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(
    args: TrainArgs,
) -> tuple[GaussianParamNetwork, OutputHead, CovariancePriorBank, "PixelResidualHead | None"]:
    """Wire up Sprint 4 components.

    Note: per-tier K-per-tile is fixed by TIER_CONFIGS in param_net.py.
    Returns ``(net, head, bank, residual_head)`` where ``residual_head`` is
    ``None`` unless ``args.enable_pixel_residual`` is set.
    """
    bank = CovariancePriorBank(learnable=False)
    net = param_net_for_tier(args.tier, bank_size=args.bank_size)
    head = OutputHead(
        bank=bank,
        k_per_tile=net.k_per_tile,
        enable_gbuffer_bias=args.enable_gbuffer_bias,
    )
    residual_head = (
        PixelResidualHead(in_channels=6, hidden_channels=args.pixel_residual_hidden)
        if args.enable_pixel_residual
        else None
    )
    return net, head, bank, residual_head


def build_sr_model_from_args(args: TrainArgs) -> "torch.nn.Module":
    """Build an SR model (SRCNNSimple or SRRRDB) from trainer args.

    Returns the model on CPU; caller is responsible for ``.to(args.device)``.
    Only called when ``args.model_kind in ("sr_cnn", "sr_rrdb")``.
    """
    from oss.sr import build_sr_model

    # model_kind selects the training target; sr_backbone selects the variant.
    # --model=sr_cnn -> kind="simple", --model=sr_rrdb -> kind="rrdb".
    # The --sr-backbone flag further overrides for the sr_cnn case.
    if args.model_kind == "sr_rrdb":
        kind = "rrdb"
    else:
        # sr_cnn: use --sr-backbone to choose simple vs rrdb
        kind = args.sr_backbone

    return build_sr_model(kind, args.tier, in_channels=12, scale=2)


@torch.no_grad()
def _sr_diagnostics(sr_model: "torch.nn.Module", out: torch.Tensor) -> dict[str, float]:
    """Diagnostic stats for the SR training path.

    Replaces the Gaussian-specific bank_entropy/dxy/color_std metrics.
    Reports model output statistics and gradient norms of first and last conv.
    """
    result: dict[str, float] = {}
    result["sr_out_mean"] = float(out.mean().item())
    result["sr_out_std"] = float(out.std().item())

    # First conv gradient norm (head_conv in SRCNNSimple, head_conv in SRRRDB).
    first_conv = getattr(sr_model, "head_conv", None)
    if first_conv is not None and first_conv.weight.grad is not None:
        result["sr_head_conv_grad_norm"] = float(
            first_conv.weight.grad.detach().norm().item()
        )
    else:
        result["sr_head_conv_grad_norm"] = 0.0

    # Last upsample conv gradient norm.
    upsample_conv = getattr(sr_model, "upsample_conv", None)
    if upsample_conv is not None and upsample_conv.weight.grad is not None:
        result["sr_upsample_conv_grad_norm"] = float(
            upsample_conv.weight.grad.detach().norm().item()
        )
    else:
        result["sr_upsample_conv_grad_norm"] = 0.0

    return result


_LPIPS_FN = None
_LPIPS_TRIED = False


def _get_lpips_fn(device: str):
    """Lazy-load the LPIPS-VGG perceptual metric.

    Returns the cached `lpips.LPIPS(net='vgg')` instance, or None when the
    package isn't importable (CI). Lower LPIPS = better perceptual quality.
    """
    global _LPIPS_FN, _LPIPS_TRIED
    if _LPIPS_TRIED:
        return _LPIPS_FN
    _LPIPS_TRIED = True
    try:
        import lpips  # type: ignore[import-not-found]
        _LPIPS_FN = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in _LPIPS_FN.parameters():
            p.requires_grad_(False)
    except ImportError:
        _LPIPS_FN = None
    return _LPIPS_FN


def _lpips(pred: torch.Tensor, target: torch.Tensor, device: str) -> float | None:
    """Compute LPIPS-VGG between two HR images in [0, 1].

    Returns None when the lpips package isn't installed.
    Inputs are expected (3, H, W) or (1, 3, H, W) in [0, 1].
    """
    fn = _get_lpips_fn(device)
    if fn is None:
        return None
    p = pred if pred.dim() == 4 else pred.unsqueeze(0)
    t = target if target.dim() == 4 else target.unsqueeze(0)
    # LPIPS expects inputs scaled to [-1, 1].
    p = (p.clamp(0.0, 1.0) * 2 - 1).to(device)
    t = (t.clamp(0.0, 1.0) * 2 - 1).to(device)
    with torch.no_grad():
        return float(fn(p, t).mean().item())


def evaluate_against_bicubic_sr(
    sr_model: "torch.nn.Module",
    dataloader,  # type: ignore[type-arg]
    device: str,
    n_samples: int = 8,
) -> dict:
    """Compare SR model PSNR + LPIPS against bicubic baseline on held-out
    examples.

    Same return-dict schema as ``evaluate_against_bicubic`` (Gaussian path)
    for consistent logging, plus optional LPIPS fields when ``lpips`` is
    importable.

    Args:
        sr_model:   Trained SR model (SRCNNSimple or SRRRDB).
        dataloader: DataLoader yielding collated GaussianTrainingExample dicts.
        device:     torch device string.
        n_samples:  Maximum examples to evaluate.

    Returns dict with keys:
        model_psnr_mean, bicubic_psnr_mean, model_psnr_per_sample,
        bicubic_psnr_per_sample, model_beats_bicubic_count,
        model_lpips_mean (or None), bicubic_lpips_mean (or None),
        model_beats_bicubic_lpips_count (or None).
    """
    sr_model.train(False)
    model_psnrs: list[float] = []
    bicubic_psnrs: list[float] = []
    model_lpips: list[float] = []
    bicubic_lpips: list[float] = []
    has_lpips = _get_lpips_fn(device) is not None

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

            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)
            H_hr, W_hr = gt_hr.shape[-2:]

            bicubic_hr = F.interpolate(
                lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)

            final = sr_model(x).clamp(0.0, 1.0)

            for b_idx in range(lr.shape[0]):
                if len(model_psnrs) >= n_samples:
                    break
                model_psnrs.append(_psnr(final[b_idx], gt_hr[b_idx]))
                bicubic_psnrs.append(_psnr(bicubic_hr[b_idx], gt_hr[b_idx]))
                if has_lpips:
                    m = _lpips(final[b_idx], gt_hr[b_idx], device)
                    b = _lpips(bicubic_hr[b_idx], gt_hr[b_idx], device)
                    if m is not None and b is not None:
                        model_lpips.append(m)
                        bicubic_lpips.append(b)

    sr_model.train(True)

    beats_count = sum(1 for m, b in zip(model_psnrs, bicubic_psnrs) if m > b)
    if model_psnrs:
        model_mean = float(sum(model_psnrs) / len(model_psnrs))
        bicubic_mean = float(sum(bicubic_psnrs) / len(bicubic_psnrs))
    else:
        model_mean = float("nan")
        bicubic_mean = float("nan")

    result: dict = {
        "model_psnr_mean": model_mean,
        "bicubic_psnr_mean": bicubic_mean,
        "model_psnr_per_sample": model_psnrs,
        "bicubic_psnr_per_sample": bicubic_psnrs,
        "model_beats_bicubic_count": beats_count,
    }
    if model_lpips:
        result["model_lpips_mean"] = float(sum(model_lpips) / len(model_lpips))
        result["bicubic_lpips_mean"] = float(sum(bicubic_lpips) / len(bicubic_lpips))
        result["model_lpips_per_sample"] = model_lpips
        result["bicubic_lpips_per_sample"] = bicubic_lpips
        # Lower LPIPS is better, so model "beats" bicubic when its LPIPS is lower.
        result["model_beats_bicubic_lpips_count"] = sum(
            1 for m, b in zip(model_lpips, bicubic_lpips) if m < b
        )
    return result


@torch.no_grad()
def _param_health(net: GaussianParamNetwork, head: OutputHead) -> dict[str, float]:
    """Sanity stats on the parameters themselves — detects 'optimiser is moving
    but you're measuring the wrong place' versus 'optimiser actually frozen'.
    """
    out = {}
    out["head_bias_abs_mean"] = float(net.head.bias.detach().abs().mean().item())
    out["head_weight_abs_mean"] = float(net.head.weight.detach().abs().mean().item())
    if hasattr(net.head.bias, "grad") and net.head.bias.grad is not None:
        out["head_bias_grad_norm"] = float(net.head.bias.grad.detach().norm().item())
    else:
        out["head_bias_grad_norm"] = 0.0
    if hasattr(net.head.weight, "grad") and net.head.weight.grad is not None:
        out["head_weight_grad_norm"] = float(net.head.weight.grad.detach().norm().item())
    else:
        out["head_weight_grad_norm"] = 0.0
    return out


@torch.no_grad()
def _compute_diagnostics(
    head: OutputHead,
    raw: torch.Tensor,
    depth: torch.Tensor | None,
    normals: torch.Tensor | None,
) -> dict[str, float]:
    """Diagnostic metrics from one forward pass — exposes whether the model
    has collapsed to a degenerate solution (constant-gray output).

    Returns:
        bank_entropy_norm: H(bank_weights) / log(bank_size) ∈ [0, 1].
                          1.0 = uniform across all entries (no learning yet).
                          ~0.0 = collapsed to a single entry.
        mean_dxy_norm:    mean |xy − tile_center| / tile_size ∈ [0, 1].
                          ~0.0 = positions stuck at tile centers.
        mean_color:       per-channel mean of decoded color in [0, 1].
                          ~0.5 across all channels = sigmoid trapped at zero
                          (constant gray).
        color_std:        std of color across all Gaussians. Low = output
                          is uniform; high = some texture variety.
    """
    decoded = head.decode(raw, depth=depth, normals=normals)
    bank_w = decoded.bank_weights  # (B, N, K_bank)
    eps = 1e-12
    H_per = -(bank_w * (bank_w.clamp(min=eps).log())).sum(dim=-1)  # (B, N)
    bank_entropy_norm = float(H_per.mean().item() / math.log(bank_w.shape[-1]))

    # Position deviation from tile center.
    xy = decoded.xy  # (B, N, 2) in pixel space
    B, N, _ = xy.shape
    K = head.k_per_tile
    tile_size = float(head.tile_size)
    Ht = raw.shape[-2]
    Wt = raw.shape[-1]
    # Reconstruct tile centers in same order decode produced them.
    ys = (torch.arange(Ht, device=xy.device, dtype=xy.dtype) + 0.5) * tile_size
    xs = (torch.arange(Wt, device=xy.device, dtype=xy.dtype) + 0.5) * tile_size
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")  # (Ht, Wt)
    centers = torch.stack([cx, cy], dim=-1)         # (Ht, Wt, 2)
    centers = centers[None, :, :, None, :].expand(B, Ht, Wt, K, 2).reshape(B, N, 2)
    dxy = (xy - centers).norm(dim=-1) / tile_size   # (B, N)
    mean_dxy_norm = float(dxy.mean().item())

    feat = decoded.feat  # (B, N, 3) in [0, 1] after sigmoid
    mean_color = feat.mean(dim=(0, 1)).tolist()
    color_std = float(feat.std().item())

    return {
        "bank_entropy_norm": bank_entropy_norm,
        "mean_dxy_norm": mean_dxy_norm,
        "mean_color_r": mean_color[0],
        "mean_color_g": mean_color[1],
        "mean_color_b": mean_color[2],
        "color_std": color_std,
    }


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
        enable_jitter=True,
        enable_taa_blur=True,
        enable_jpeg=args.lr_synth_jpeg,
        jpeg_quality=args.lr_synth_jpeg_quality,
        blur_sigma=args.lr_synth_blur_sigma,
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
    renderer_backend: str = "auto",
    residual_head: "PixelResidualHead | None" = None,
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
    renderer = Rasterizer(
        force_backend=None if renderer_backend == "auto" else renderer_backend
    )
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

                # Apply V0.5 pixel-residual head if wired.
                if residual_head is not None:
                    res = residual_head(
                        rendered.unsqueeze(0),
                        bicubic_hr[b_idx : b_idx + 1],
                    ).squeeze(0)
                    rendered = (rendered + res).clamp(0.0, 1.0)

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
    net: "GaussianParamNetwork | None",
    bank: "CovariancePriorBank | None",
    args: TrainArgs,
    residual_head: "PixelResidualHead | None" = None,
    sr_model: "torch.nn.Module | None" = None,
    optim: "torch.optim.Optimizer | None" = None,
) -> None:
    """Save a training checkpoint.

    For the Gaussian track (``args.model_kind == "gaussian"``), saves ``net`` +
    ``bank`` + optional ``residual_head``.  For the SR track, saves
    ``sr_model`` under the ``"sr_model"`` key.  Both paths write the full
    ``args`` dict for reproducibility.

    When ``optim`` is provided, its ``state_dict`` is also saved so that
    auto-resume can restore AdamW momentum/variance and the run continues
    seamlessly after a process death.
    """
    ckpt_path = output_dir / f"step-{step:08d}.pt"
    payload: dict = {
        "step": step,
        "tier": tier,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in args.__dict__.items()
        },
    }
    if optim is not None:
        payload["optim"] = optim.state_dict()
    if sr_model is not None:
        # SR track: only the SR model state is needed.
        payload["sr_model"] = sr_model.state_dict()
        payload["model_kind"] = args.model_kind
    else:
        # Gaussian track: net + bank + optional residual head.
        payload["net"] = net.state_dict()
        payload["bank"] = bank.state_dict()
        if residual_head is not None:
            payload["residual_head"] = residual_head.state_dict()
    torch.save(payload, ckpt_path)
    log.info("ckpt -> %s", ckpt_path)


# ---------------------------------------------------------------------------
# SR training loop (2026-05-02 pivot — bypasses splat rendering)
# ---------------------------------------------------------------------------


def _main_sr(args: TrainArgs) -> int:
    """Training loop for the SR CNN track (SRCNNSimple or SRRRDB).

    Structurally mirrors the Gaussian real-data loop but:
    - Builds the SR model (no net/head/bank/renderer).
    - Calls sr_model(x).clamp(0, 1) directly — no splat rendering.
    - Skips Gaussian-specific diagnostics (bank_entropy, dxy, color_std, etc.)
      and replaces them with SR output stats + conv grad norms.
    - Saves checkpoints under the "sr_model" key.

    The bicubic comparison uses evaluate_against_bicubic_sr() which has the
    same return schema as evaluate_against_bicubic() for consistent logging.
    """
    sr_model = build_sr_model_from_args(args)
    sr_model.to(args.device)

    optim = torch.optim.AdamW(
        sr_model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    log.info(
        "SR model: kind=%s backbone=%s tier=%s params=%d",
        args.model_kind,
        args.sr_backbone,
        args.tier,
        sum(p.numel() for p in sr_model.parameters()),
    )

    metrics_log: list[dict] = []
    score_log: list[dict] = []
    train_start = time.monotonic()
    resume_step = 0

    # Auto-resume from the most recent checkpoint in args.output_dir if any.
    # This protects 36+hr runs from process death — restart with the same CLI
    # and training picks up from the last 5000-step boundary.
    if args.output_dir.exists():
        ckpts = sorted(args.output_dir.glob("step-*.pt"))
        if ckpts:
            latest = ckpts[-1]
            log.info("SR: resuming from %s", latest)
            ck = torch.load(latest, map_location=args.device, weights_only=False)
            if "sr_model" in ck:
                sr_model.load_state_dict(ck["sr_model"])
            if "optim" in ck:
                optim.load_state_dict(ck["optim"])
            resume_step = int(ck.get("step", 0))
            # Restore previously-flushed metrics if present (avoids losing history).
            mp = args.output_dir / "metrics.json"
            if mp.exists():
                with mp.open() as _f:
                    _saved = json.load(_f)
                metrics_log = _saved.get("train", [])
                score_log = _saved.get("score", [])
            sp = args.output_dir / "score_log.json"
            if sp.exists() and not score_log:
                with sp.open() as _f:
                    score_log = json.load(_f)
            log.info("SR: resumed at step=%d (metrics=%d entries, score=%d entries)",
                     resume_step, len(metrics_log), len(score_log))

    # ------------------------------------------------------------------
    # Synthetic-batch path (CI / sanity -- no real data needed)
    # ------------------------------------------------------------------
    if args.use_synthetic_batch:
        h, w = 64, 64
        log.info("SR: using synthetic_batch path (no real data)")
        final_step = 0
        for step in range(resume_step + 1, args.max_steps + 1):
            if args.max_time_seconds is not None:
                elapsed = time.monotonic() - train_start
                if elapsed > args.max_time_seconds:
                    log.info("SR: wall-clock limit at step %d (%.1f s)", step, elapsed)
                    step -= 1
                    break

            x, target = synthetic_batch(args.batch_size, h, w, args.device)
            optim.zero_grad()
            # No clamp during training — clamp(0,1) creates a gradient-killer
            # whenever (bicubic + residual) leaves [0, 1], which happens at
            # standard tier on multi-scene data even with depth-aware init.
            # Loss naturally penalises out-of-range output via L1.
            final = sr_model(x)
            loss, parts = composite_loss(
                final, target,
                w_lpips=args.lpips_loss_weight,
                lpips_device=args.device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sr_model.parameters(), max_norm=1.0)
            optim.step()

            if step % args.log_every == 0:
                diag = _sr_diagnostics(sr_model, final.detach())
                row = {"step": step, "loss": float(loss.item()), **parts, **diag}
                metrics_log.append(row)
                aux_key = "ssim" if "ssim" in row else "pooled_l1"
                log.info(
                    "SR step=%d loss=%.4f l1=%.4f %s=%.4f out_mean=%.3f out_std=%.3f",
                    step, row["loss"], row["l1"], aux_key, row[aux_key],
                    row["sr_out_mean"], row["sr_out_std"],
                )

            if step % args.ckpt_every == 0 or step == args.max_steps:
                _save_checkpoint(
                    args.output_dir, step, args.tier,
                    net=None, bank=None, args=args, sr_model=sr_model, optim=optim,  # type: ignore[arg-type]
                )

            final_step = step

    # ------------------------------------------------------------------
    # Real-data path (SR track)
    # ------------------------------------------------------------------
    else:
        loader = build_dataloader(args)
        score_loader = build_dataloader(args)
        log.info(
            "SR: dataset size=%d sequence_filter=%r",
            len(loader.dataset),  # type: ignore[arg-type]
            args.sintel_sequence,
        )

        step = resume_step
        final_step = resume_step
        data_iter = iter(loader)

        while step < args.max_steps:
            if args.max_time_seconds is not None:
                elapsed = time.monotonic() - train_start
                if elapsed > args.max_time_seconds:
                    log.info("SR: wall-clock limit at step %d (%.1f s)", step, elapsed)
                    break

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

            # 12-channel input: LR(3)+depth(1)+motion(2)+normals(3)+canvas(3).
            x = torch.cat([lr, depth, motion, normals, canvas], dim=1)

            optim.zero_grad()
            # No clamp during training (see synth-batch path for rationale).
            final = sr_model(x)
            loss, parts = composite_loss(
                final, gt_hr,
                w_lpips=args.lpips_loss_weight,
                lpips_device=args.device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sr_model.parameters(), max_norm=1.0)
            optim.step()

            if step % args.log_every == 0:
                diag = _sr_diagnostics(sr_model, final.detach())
                row = {"step": step, "loss": float(loss.item()), **parts, **diag}
                metrics_log.append(row)
                aux_key = "ssim" if "ssim" in row else "pooled_l1"
                lpips_str = f" lpips={row['lpips']:.4f}" if "lpips" in row else ""
                log.info(
                    "SR step=%d loss=%.4f l1=%.4f %s=%.4f%s out_mean=%.3f out_std=%.3f "
                    "head_grad=%.4e up_grad=%.4e",
                    step, row["loss"], row["l1"], aux_key, row[aux_key], lpips_str,
                    row["sr_out_mean"], row["sr_out_std"],
                    row["sr_head_conv_grad_norm"], row["sr_upsample_conv_grad_norm"],
                )

            if step % args.ckpt_every == 0:
                _save_checkpoint(
                    args.output_dir, step, args.tier,
                    net=None, bank=None, args=args, sr_model=sr_model, optim=optim,  # type: ignore[arg-type]
                )
                # Rolling metrics dump — survives process death.
                _metrics_path = args.output_dir / "metrics.json"
                with _metrics_path.open("w") as _f:
                    json.dump({"train": metrics_log, "score": score_log}, _f, indent=2)

            if args.score_every > 0 and step % args.score_every == 0:
                log.info("SR: bicubic comparison at step %d", step)
                result = evaluate_against_bicubic_sr(
                    sr_model, score_loader, args.device, n_samples=8
                )
                score_row = {"step": step, **result}
                score_log.append(score_row)
                # Rolling score dump after each eval — survives process death.
                _score_path = args.output_dir / "score_log.json"
                with _score_path.open("w") as _f:
                    json.dump(score_log, _f, indent=2)
                if "model_lpips_mean" in result:
                    log.info(
                        "SR step=%d model_psnr=%.2f dB  bicubic_psnr=%.2f dB  "
                        "beats_bicubic=%d/8  "
                        "model_lpips=%.4f bicubic_lpips=%.4f beats_lpips=%d/8",
                        step,
                        result["model_psnr_mean"],
                        result["bicubic_psnr_mean"],
                        result["model_beats_bicubic_count"],
                        result["model_lpips_mean"],
                        result["bicubic_lpips_mean"],
                        result["model_beats_bicubic_lpips_count"],
                    )
                else:
                    log.info(
                        "SR step=%d model_psnr=%.2f dB  bicubic_psnr=%.2f dB  "
                        "beats_bicubic=%d/8",
                        step,
                        result["model_psnr_mean"],
                        result["bicubic_psnr_mean"],
                        result["model_beats_bicubic_count"],
                    )

        # Final checkpoint + comparison.
        _save_checkpoint(
            args.output_dir, final_step, args.tier,
            net=None, bank=None, args=args, sr_model=sr_model, optim=optim,  # type: ignore[arg-type]
        )

        log.info("SR: final bicubic comparison at step %d", final_step)
        final_result = evaluate_against_bicubic_sr(
            sr_model, score_loader, args.device, n_samples=8
        )
        score_log.append({"step": final_step, "final": True, **final_result})
        log.info(
            "SR FINAL model_psnr=%.2f dB  bicubic_psnr=%.2f dB  beats_bicubic=%d/8",
            final_result["model_psnr_mean"],
            final_result["bicubic_psnr_mean"],
            final_result["model_beats_bicubic_count"],
        )

    # Write metrics.
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"train": metrics_log, "score": score_log}, f, indent=2)
    log.info("SR metrics -> %s", metrics_path)

    elapsed_total = time.monotonic() - train_start
    log.info("SR done: steps=%d elapsed=%.1f s", final_step, elapsed_total)
    return 0


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

    # ---------------------------------------------------------------------------
    # SR-track dispatch (2026-05-02 pivot).  The Gaussian path below is
    # bit-identical to the pre-pivot behavior when model_kind == "gaussian".
    # ---------------------------------------------------------------------------
    if args.model_kind in ("sr_cnn", "sr_rrdb"):
        return _main_sr(args)

    net, head, bank, residual_head = build_model(args)
    net.to(args.device)
    bank.to(args.device)
    if args.enable_gbuffer_bias and head.gbuffer_bias is not None:
        head.gbuffer_bias.to(args.device)
    if residual_head is not None:
        residual_head.to(args.device)

    renderer = Rasterizer(
        force_backend=None if args.renderer_backend == "auto" else args.renderer_backend
    )
    optim_params = list(net.parameters()) + list(bank.parameters())
    if residual_head is not None:
        optim_params += list(residual_head.parameters())
    optim = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=1e-5)
    log.info(
        "net params=%d bank params=%d residual params=%d",
        sum(p.numel() for p in net.parameters()),
        sum(p.numel() for p in bank.parameters()),
        sum(p.numel() for p in residual_head.parameters()) if residual_head else 0,
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
                diag = _compute_diagnostics(head, raw, depth=None, normals=None)
                row = {"step": step, "loss": float(loss.item()), **parts, **diag}
                metrics_log.append(row)
                aux_key = "ssim" if "ssim" in row else "pooled_l1"
                log.info(
                    "step=%d loss=%.4f l1=%.4f %s=%.4f bank_H=%.3f dxy=%.3f color_std=%.3f",
                    step, row["loss"], row["l1"], aux_key, row[aux_key],
                    row["bank_entropy_norm"], row["mean_dxy_norm"], row["color_std"],
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

            # V0.5: pixel-residual head on top of the splat raster.
            if residual_head is not None:
                lr_up = F.interpolate(
                    lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
                ).clamp(0.0, 1.0)
                residual = residual_head(rendered.clamp(0.0, 1.0), lr_up)
                final = (rendered + residual).clamp(0.0, 1.0)
            else:
                final = rendered

            loss, parts = composite_loss(
                final, gt_hr,
                w_lpips=args.lpips_loss_weight,
                lpips_device=args.device,
            )
            loss.backward()
            clip_params = list(net.parameters()) + list(bank.parameters())
            if residual_head is not None:
                clip_params += list(residual_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            optim.step()

            if step % args.log_every == 0:
                diag = _compute_diagnostics(head, raw, depth=depth, normals=normals)
                health = _param_health(net, head)
                row = {"step": step, "loss": float(loss.item()), **parts, **diag, **health}
                metrics_log.append(row)
                aux_key = "ssim" if "ssim" in row else "pooled_l1"
                log.info(
                    "step=%d loss=%.4f l1=%.4f %s=%.4f bank_H=%.3f dxy=%.3f cstd=%.3f bias_abs=%.4f bias_grad=%.4e w_grad=%.4e",
                    step, row["loss"], row["l1"], aux_key, row[aux_key],
                    row["bank_entropy_norm"], row["mean_dxy_norm"], row["color_std"],
                    row["head_bias_abs_mean"], row["head_bias_grad_norm"], row["head_weight_grad_norm"],
                )

            if step % args.ckpt_every == 0:
                _save_checkpoint(args.output_dir, step, args.tier, net, bank, args, residual_head=residual_head)

            # Periodic bicubic comparison.
            if args.score_every > 0 and step % args.score_every == 0:
                log.info("--- bicubic comparison at step %d ---", step)
                result = evaluate_against_bicubic(
                    net, head, bank, score_loader, args.device, n_samples=8,
                    renderer_backend=args.renderer_backend,
                    residual_head=residual_head,
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
        _save_checkpoint(args.output_dir, final_step, args.tier, net, bank, args, residual_head=residual_head)

        # Final bicubic comparison (always at end of real-data training).
        log.info("--- final bicubic comparison at step %d ---", final_step)
        final_result = evaluate_against_bicubic(
            net, head, bank, score_loader, args.device, n_samples=8,
            renderer_backend=args.renderer_backend,
            residual_head=residual_head,
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

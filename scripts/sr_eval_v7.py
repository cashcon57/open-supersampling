"""v7 Phase 3 eval script.

Loads a checkpoint produced by ``scripts/sr_train_v7.py`` and reports
PSNR / SSIM / LPIPS-VGG at BOTH:

  - alpha = 1   (frame N+1 SR -- the classic SR job)
  - alpha = 0.5 (intermediate frame N + 0.5 -- the OSS-FX claim)

Plus a bicubic-midpoint baseline on the alpha = 0.5 case so we can read
off the pass criterion from the Phase 3 plan:

    alpha=0.5 PSNR >= bicubic_midpoint_alpha=0.5 PSNR + 1.0 dB

Per-triplet forward flow mirrors the trainer exactly:

    model.reset_state(device)
    _      = model(n_lr_in,   t_query=0.0, spawn_at_t=0.0)   # warmup, not scored
    out_sr = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)   # alpha=1 SR
    out_fx = model(np1_lr_in, t_query=1.0)                   # alpha=0.5 OSS-FX

GT is clamped to [0, 1] to match the trainer.

Output: ``<output-dir>/eval-step-NNNNNNNN.json`` with the structure
shown in docs/architecture/2026-05-12-v7-pico-005-phase-3-plan.md.

Usage:
    python scripts/sr_eval_v7.py \\
        --checkpoint E:/checkpoints/srcnn-v7.0-pico-005/step-00100000.pt \\
        --tartanair-root E:/datasets/tartanair_extracted \\
        --output-dir E:/checkpoints/srcnn-v7.0-pico-005/eval \\
        --max-triplets 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from oss.sr.v7.model import V7Config, V7Model
from oss.sr.v7.intermediate_dataset import TartanAirIntermediateTriplets


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


def _device(arg: str) -> str:
    if arg == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA unavailable, falling back to CPU.")
        return "cpu"
    return arg


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR with the Phase-3-plan formula: 20*log10(1/sqrt(mse)). Inputs
    are clamped to [0, 1] (data_range = 1).
    """
    pred_c = pred.clamp(0.0, 1.0).float()
    gt_c = gt.clamp(0.0, 1.0).float()
    mse = float(((pred_c - gt_c) ** 2).mean().item())
    if mse <= 0.0:
        # All-identical case -- return a large finite cap rather than
        # +inf so downstream JSON serialization stays usable.
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


class _LazySSIM:
    _fn = None
    _available: Optional[bool] = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            try:
                from pytorch_msssim import ssim as _ssim_fn  # type: ignore[import-not-found]

                cls._fn = _ssim_fn
                cls._available = True
            except Exception:
                cls._fn = None
                cls._available = False
        return cls._available

    @classmethod
    def score(cls, pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
        if not cls.available():
            return None
        p = pred.clamp(0.0, 1.0).float()
        g = gt.clamp(0.0, 1.0).float()
        if p.dim() == 3:
            p = p.unsqueeze(0)
            g = g.unsqueeze(0)
        return float(cls._fn(p, g, data_range=1.0, size_average=True).item())


class _LazyLPIPS:
    """LPIPS-VGG with the same lazy-load contract as ``oss/sr/v7/losses.py``."""

    _instance = None  # None = not tried, False = unavailable, object = ready

    @classmethod
    def get(cls, device):
        if cls._instance is False:
            return None
        if cls._instance is None:
            try:
                import lpips  # type: ignore[import-not-found]

                m = lpips.LPIPS(net="vgg", verbose=False).to(device)
                m.train(False)
                for p in m.parameters():
                    p.requires_grad_(False)
                cls._instance = m
            except Exception as e:  # pragma: no cover - exercised on hosts w/o lpips
                print(f"[eval] LPIPS-VGG unavailable ({e}); LPIPS metrics will be null")
                cls._instance = False
                return None
        return cls._instance

    @classmethod
    def score(cls, pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
        m = cls.get(pred.device)
        if m is None:
            return None
        p = pred.clamp(0.0, 1.0).float()
        g = gt.clamp(0.0, 1.0).float()
        if p.dim() == 3:
            p = p.unsqueeze(0)
            g = g.unsqueeze(0)
        # lpips expects inputs in [-1, 1]
        return float(m(p * 2.0 - 1.0, g * 2.0 - 1.0).mean().item())


def _bicubic_midpoint(
    n_lr: torch.Tensor, np1_lr: torch.Tensor, output_hw: tuple[int, int]
) -> torch.Tensor:
    """True midpoint baseline at alpha=0.5: pixel-averaged bicubic of BOTH
    endpoints upsampled to HR. This is the dumbest "predict the midpoint
    frame given frame N and frame N+1" baseline you can write -- it's the
    floor v7's OSS-FX has to clear.

    Earlier versions of this function used only `np1_lr` (right endpoint
    only), which is an asymmetric baseline and biases the +1 dB pass
    criterion toward whichever endpoint is closer to the held-out half
    frame. Using the average is symmetric in time and matches the
    "naive frame-interp from neighbors" intuition that the alpha=0.5
    metric is supposed to beat.

    Returns (B, 3, H_hr, W_hr) clamped to [0, 1].
    """
    def _up(x):
        return F.interpolate(
            x[:, :3], size=output_hw, mode="bicubic",
            antialias=True, align_corners=False,
        )
    return (0.5 * (_up(n_lr) + _up(np1_lr))).clamp(0.0, 1.0)


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------


def _load_checkpoint(ckpt_path: Path, device: str):
    """Load a v7 checkpoint and rebuild the V7Model. Returns (model, step, meta).

    The checkpoint format matches ``sr_train_v7.py``:
        {"step": int, "model_state": state_dict, "cfg": vars(V7Config), "args": ...}
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f"{ckpt_path}: expected dict checkpoint, got {type(ck)}")

    cfg_data = ck.get("cfg", {}) or {}
    valid_keys = {f.name for f in fields(V7Config)}
    cfg_kwargs = {k: v for k, v in cfg_data.items() if k in valid_keys}
    cfg = V7Config(**cfg_kwargs)
    model = V7Model(cfg).to(device)
    model.allocate_canvas(device)

    state = ck.get("model_state")
    if state is None:
        # Be tolerant of legacy / alt key names.
        for key in ("state_dict", "model"):
            if key in ck:
                state = ck[key]
                break
    if state is None:
        raise KeyError(f"{ckpt_path} has no model_state / state_dict key")

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"[eval] checkpoint schema drift: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )
    model.train(False)
    step = int(ck.get("step", 0))
    return model, step, ck


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------


def _build_9ch(lr: torch.Tensor, depth: torch.Tensor, motion: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    return torch.cat([lr, depth, motion, normals], dim=1).contiguous()


def _build_held_out_dataset(
    tartanair_root: Path,
    max_triplets: int,
    seed: int,
) -> TartanAirIntermediateTriplets:
    """Build a held-out triplet dataset. We pull the full set of valid
    triplets, then deterministically subsample to ``max_triplets``.
    """
    from oss.gaussian.data import TartanAirGaussianDataset

    base = TartanAirGaussianDataset(root=tartanair_root, scale=2.0)
    # Build the full triplet index first so subsampling spans the whole
    # held-out trajectory rather than just the first N consecutive
    # frames.
    ds = TartanAirIntermediateTriplets(base, max_triplets=None)
    total = len(ds)
    if total == 0:
        raise RuntimeError(f"No valid triplets under {tartanair_root}")
    rng = random.Random(seed)
    if total <= max_triplets:
        sel = list(range(total))
    else:
        sel = sorted(rng.sample(range(total), max_triplets))
    # Mutate the triplet index list in place so __len__/__getitem__ map
    # onto our subset. This is cheaper than wrapping in a Subset.
    ds._triplet_indices = [ds._triplet_indices[i] for i in sel]
    return ds


# ---------------------------------------------------------------------
# Canvas-health snapshot (mirrors sr_train_v7.canvas_health_metrics)
# ---------------------------------------------------------------------


def _canvas_health(model: V7Model) -> dict[str, float]:
    cs = model.canvas
    n_active = int(cs.count)
    if n_active == 0:
        return {"count": 0, "mean_opacity": 0.0, "mean_L_diag": 0.0}
    live_mask = cs.mask[: cs.n_live]
    idx = live_mask.nonzero(as_tuple=True)[0]
    opacity = cs.opacity[: cs.n_live][idx]
    cov_raw = cs.cov_raw[: cs.n_live][idx]
    L_diag = torch.stack(
        [cov_raw[:, 0].exp(), cov_raw[:, 2].exp(), cov_raw[:, 5].exp()], dim=-1
    )
    return {
        "count": n_active,
        "mean_opacity": float(opacity.mean().item()),
        "mean_L_diag": float(L_diag.mean().item()),
    }


# ---------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------


def evaluate(
    model: V7Model,
    dataset,
    device: str,
) -> dict[str, Any]:
    """Run the eval. Returns a dict ready to be JSON-serialized.

    ``dataset`` must be indexable and produce the triplet dict shape
    that ``TartanAirIntermediateTriplets`` emits.
    """
    psnr_sr: list[float] = []
    ssim_sr: list[Optional[float]] = []
    lpips_sr: list[Optional[float]] = []
    psnr_fx: list[float] = []
    ssim_fx: list[Optional[float]] = []
    lpips_fx: list[Optional[float]] = []
    psnr_bi: list[float] = []
    ssim_bi: list[Optional[float]] = []
    lpips_bi: list[Optional[float]] = []

    model.train(False)
    last_health = {"count": 0, "mean_opacity": 0.0, "mean_L_diag": 0.0}

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            n_lr_in = _build_9ch(
                sample["n"]["lr"].unsqueeze(0).to(device),
                sample["n"]["depth"].unsqueeze(0).to(device),
                sample["n"]["motion"].unsqueeze(0).to(device),
                sample["n"]["normals"].unsqueeze(0).to(device),
            )
            np1_lr_in = _build_9ch(
                sample["n_plus_1"]["lr"].unsqueeze(0).to(device),
                sample["n_plus_1"]["depth"].unsqueeze(0).to(device),
                sample["n_plus_1"]["motion"].unsqueeze(0).to(device),
                sample["n_plus_1"]["normals"].unsqueeze(0).to(device),
            )
            n_half_gt = sample["n_half"]["gt_hr"].unsqueeze(0).to(device).clamp(0.0, 1.0)
            np1_gt = sample["n_plus_1"]["gt_hr"].unsqueeze(0).to(device).clamp(0.0, 1.0)

            # Mirror trainer flow exactly.
            model.reset_state(device)
            _ = model(n_lr_in, t_query=0.0, spawn_at_t=0.0)
            out_main_np1 = model(np1_lr_in, t_query=2.0, spawn_at_t=2.0)
            out_inter = model(np1_lr_in, t_query=1.0)

            # alpha = 1 SR metrics
            psnr_sr.append(_psnr(out_main_np1, np1_gt))
            ssim_sr.append(_LazySSIM.score(out_main_np1, np1_gt))
            lpips_sr.append(_LazyLPIPS.score(out_main_np1, np1_gt))

            # alpha = 0.5 OSS-FX metrics
            psnr_fx.append(_psnr(out_inter, n_half_gt))
            ssim_fx.append(_LazySSIM.score(out_inter, n_half_gt))
            lpips_fx.append(_LazyLPIPS.score(out_inter, n_half_gt))

            # Bicubic-midpoint baseline at alpha = 0.5: pixel-average of
            # bicubic-upsampled frame N and frame N+1 LR -- the symmetric
            # naive "predict the in-between frame" baseline that v7's
            # OSS-FX has to beat by >=1 dB to clear the Phase 3 floor.
            output_hw = (n_half_gt.shape[-2], n_half_gt.shape[-1])
            bi = _bicubic_midpoint(n_lr_in, np1_lr_in, output_hw)
            psnr_bi.append(_psnr(bi, n_half_gt))
            ssim_bi.append(_LazySSIM.score(bi, n_half_gt))
            lpips_bi.append(_LazyLPIPS.score(bi, n_half_gt))

            last_health = _canvas_health(model)

    def _mean(xs):
        vals = [v for v in xs if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    alpha_1 = {"psnr": _mean(psnr_sr), "ssim": _mean(ssim_sr), "lpips": _mean(lpips_sr)}
    alpha_0_5 = {"psnr": _mean(psnr_fx), "ssim": _mean(ssim_fx), "lpips": _mean(lpips_fx)}
    alpha_0_5_bi = {"psnr": _mean(psnr_bi), "ssim": _mean(ssim_bi), "lpips": _mean(lpips_bi)}

    delta_db = None
    if alpha_0_5["psnr"] is not None and alpha_0_5_bi["psnr"] is not None:
        delta_db = float(alpha_0_5["psnr"] - alpha_0_5_bi["psnr"])

    result: dict[str, Any] = {
        "n_triplets": len(dataset),
        "alpha_1_sr": alpha_1,
        "alpha_0_5_oss_fx": alpha_0_5,
        "alpha_0_5_bicubic_baseline": alpha_0_5_bi,
        "delta_oss_fx_over_bicubic_psnr_db": delta_db,
        "canvas_health_final": last_health,
    }
    notes: list[str] = []
    if not _LazySSIM.available():
        notes.append("SSIM skipped: pytorch_msssim not installed")
    if _LazyLPIPS.get(torch.device("cpu")) is None:
        notes.append("LPIPS skipped: lpips package not installed")
    if notes:
        result["notes"] = notes
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def run_eval(
    checkpoint: Path,
    tartanair_root: Path,
    output_dir: Path,
    device: str = "cuda",
    max_triplets: int = 64,
    seed: int = 42,
) -> Path:
    """Programmatic entry point. Returns the path to the written JSON.

    The unit tests use this so they can construct a fake dataset and
    bypass the CLI argv parsing.
    """
    device = _device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, step, _ = _load_checkpoint(checkpoint, device)

    dataset = _build_held_out_dataset(tartanair_root, max_triplets=max_triplets, seed=seed)
    print(f"[eval] held-out triplets: {len(dataset)} (cap={max_triplets}, seed={seed})")

    result = evaluate(model, dataset, device)
    result["checkpoint"] = str(checkpoint)
    result["step"] = step

    out_json = output_dir / f"eval-step-{step:08d}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    _print_summary(result)
    print(f"[eval] wrote {out_json}")
    return out_json


def run_eval_with_dataset(
    checkpoint: Path,
    dataset,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    """Same as ``run_eval`` but takes a pre-built dataset. Used by the
    tests to inject a synthetic triplet source without TartanAir on
    disk.
    """
    device = _device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, step, _ = _load_checkpoint(checkpoint, device)
    result = evaluate(model, dataset, device)
    result["checkpoint"] = str(checkpoint)
    result["step"] = step
    out_json = output_dir / f"eval-step-{step:08d}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    _print_summary(result)
    print(f"[eval] wrote {out_json}")
    return out_json


def _print_summary(result: dict[str, Any]) -> None:
    a1 = result["alpha_1_sr"]["psnr"]
    afx = result["alpha_0_5_oss_fx"]["psnr"]
    abi = result["alpha_0_5_bicubic_baseline"]["psnr"]
    d = result["delta_oss_fx_over_bicubic_psnr_db"]

    def _fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"

    sign = "+" if (d is not None and d >= 0) else ""
    print(
        f"[eval] α=1: PSNR={_fmt(a1)}  "
        f"α=0.5: PSNR={_fmt(afx)}  "
        f"bicubic_α=0.5: PSNR={_fmt(abi)}  "
        f"Δ={sign}{_fmt(d)} dB"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a .pt checkpoint saved by sr_train_v7.py")
    p.add_argument("--tartanair-root", type=Path, required=True,
                   help="TartanAir extracted root (same one the trainer uses)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write eval-step-NNNNNNNN.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-triplets", type=int, default=64,
                   help="Cap on held-out triplets for speed (default 64)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for deterministic held-out subset selection")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        print(f"FAIL: checkpoint not found: {args.checkpoint}")
        return 1
    run_eval(
        checkpoint=args.checkpoint,
        tartanair_root=args.tartanair_root,
        output_dir=args.output_dir,
        device=args.device,
        max_triplets=args.max_triplets,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

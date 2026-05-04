"""Held-out fixed-batch eval: v5-gaussian-temporal vs v5-pixel-temporal vs v4.

Sprint-5 ship-decision evaluator for the v5-gaussian-temporal track. Mirrors
``scripts/sr_temporal_held_out.py`` (the pixel-temporal held-out eval) but
scores THREE models on the SAME deterministic batch:

  - v5-gaussian-temporal  (G) — Task 8 model, stateful Gaussian-field engine.
  - v5-pixel-temporal     (P) — pixel-warp temporal, prev_hr cold-start regime.
  - v4 single-frame       (A) — baseline (used both as competitor and as
    cold-start prev_hr for the pixel-temporal model on the first frame).

Each held-out frame ``t+1`` is scored on:

  - PSNR (higher better)
  - LPIPS-VGG (lower better; perceptual)
  - Temporal stability: ``mean(|warp(out_t, motion_t->t+1) - out_{t+1}|_1)``
    (lower = more temporally stable)

Reports per-sample win counts:

  - ``G > P``   (gaussian beats pixel) — strict; tie != win (race rule)
  - ``G > A``   (gaussian beats baseline)
  - ``P > A``   (pixel beats baseline)
  - joint criterion vs bicubic for each of G / P.

Writes ``held_out_results.json`` next to the gaussian checkpoint.

Flow-direction convention (matches the pixel held-out fix at commit 38cf507):
    Pass ``t_motion`` (forward flow t->t+1, lives at frame t) when warping
    ``out_t`` to align with t+1, NOT ``tp1_motion`` (which is t+1->t+2).

Verification gate (Task 12 of the v5-gaussian-temporal plan)::

    python scripts/sr_gaussian_temporal_held_out.py --help

returning exit code 0. The actual eval requires real datasets and trained
checkpoints and is run after Sprint-5 training completes for both tracks.

Usage::

    python scripts/sr_gaussian_temporal_held_out.py \\
        --ckpt-gaussian <train-host-data>/checkpoints/srcnn-v5-gaussian-temporal/step-XXXXX.pt \\
        --ckpt-pixel    <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-XXXXX.pt \\
        --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \\
        --tartanair-root <train-host-data>/datasets/tartanair_extracted \\
        --sintel-root <train-host-data>/datasets/sintel \\
        --n-samples 64
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

# Allow ``python scripts/sr_gaussian_temporal_held_out.py`` to import ``oss.*``
# when the package isn't installed into the active interpreter (e.g. tests
# invoke us via ``sys.executable``). Mirrors ``scripts/sr_temporal_held_out.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# NOTE: torch and oss imports are deferred into ``main()`` and helpers below
# so that ``--help`` (the Task 12 verification gate) works on a vanilla
# Python interpreter without the heavy ML stack installed.

if TYPE_CHECKING:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader  # noqa: F401


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _psnr(pred, target) -> float:
    import torch.nn.functional as F  # local: keep --help torch-free
    mse = float(F.mse_loss(pred.float(), target.float()).item())
    mse = max(mse, 1e-12)
    return float(-10.0 * math.log10(mse))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_baseline(ckpt_path: Path, device: str):
    """Load a v4 single-frame SR-CNN checkpoint."""
    import torch
    from oss.sr import build_sr_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tier = saved_args.get("tier", "standard")
    sr_backbone = saved_args.get("sr_backbone", "simple")
    factory_kind = "rrdb" if sr_backbone == "rrdb" else "simple"
    model = build_sr_model(
        model_kind=factory_kind, tier=tier, in_channels=12, scale=2,
    ).to(device)
    model.load_state_dict(ckpt["sr_model"])
    model.train(False)
    return model


def _load_pixel_temporal(ckpt_path: Path, device: str):
    """Load a v5-pixel-temporal checkpoint (TemporalSRModel state)."""
    import torch
    from oss.sr.temporal import TemporalSRModel
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {})
    tier = saved.get("tier", "standard")
    backbone_kind = saved.get("backbone_kind", "simple")
    model = TemporalSRModel(
        in_channels=12, scale=2, tier=tier, backbone_kind=backbone_kind,
    ).to(device)
    if "temporal_model" in ck:
        model.load_state_dict(ck["temporal_model"])
    elif "sr_model" in ck:
        model.backbone.load_state_dict(ck["sr_model"])
    else:
        raise KeyError(
            f"checkpoint {ckpt_path} has neither 'temporal_model' nor "
            f"'sr_model' key (got {list(ck.keys())})"
        )
    model.train(False)
    return model


def _load_gaussian_engine(ckpt_path: Path, device: str):
    """Load a v5-gaussian-temporal stateful inference engine."""
    from oss.sr.inference import GaussianTemporalSRInferenceEngine
    # fp16=False for held-out: deterministic-friendly + cpu-compatible.
    return GaussianTemporalSRInferenceEngine.from_checkpoint(
        ckpt_path, device=device, fp16=False, scene_cut_motion_threshold=32.0,
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _build_pair_loader(kind: str, root: Path, batch_size: int):
    """Build a deterministic SequentialPairDataset loader.

    ``shuffle=False`` and zero-worker keep frame order stable across runs;
    the caller seeds ``torch.manual_seed(0)`` before iterating. Reuses the
    pixel-temporal adapters since the gaussian-temporal pipeline consumes
    the same g-buffer fields.
    """
    from torch.utils.data import DataLoader
    from oss.gaussian.data import (
        SintelGaussianDataset,
        TartanAirGaussianDataset,
    )
    from oss.sr.temporal import (
        SequentialPairDataset, adapt_sintel, adapt_tartanair,
        default_collate_pair,
    )
    if kind == "tartanair":
        ds = adapt_tartanair(TartanAirGaussianDataset(root=root, scale=2.0))
    elif kind == "sintel":
        ds = adapt_sintel(
            SintelGaussianDataset(root=root, scale=2.0, pass_name="clean")
        )
    else:
        raise ValueError(f"unknown dataset kind: {kind!r}")
    pair = SequentialPairDataset(ds)
    if len(pair) == 0:
        return None
    return DataLoader(
        pair, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=default_collate_pair, drop_last=True,
    )


# ---------------------------------------------------------------------------
# Eval core
# ---------------------------------------------------------------------------


def _make_12ch(lr, depth, motion, normals, canvas):
    import torch
    return torch.cat([lr, depth, motion, normals, canvas], dim=1)


def _eval_loader(
    loader,
    *,
    engine_gaussian,
    model_pixel,
    model_baseline,
    lpips_fn,
    n_samples_remaining: int,
    device: str,
) -> dict[str, list[float]]:
    """Run held-out eval on a single dataset loader.

    For each batch we evaluate frame ``t+1`` for the four reconstructions
    (gaussian / pixel / baseline / bicubic) plus temporal-stability metrics
    for gaussian + pixel + baseline.

    The Gaussian engine is stateful (B=1 only) so we iterate per-sample
    inside each batch, calling ``engine.reset()`` between samples to ensure
    each held-out pair is treated as a clean two-frame stream (frame t seeds
    the field; frame t+1 is the scored output).

    Returns a dict of lists, one entry per held-out frame consumed.
    """
    import torch
    import torch.nn.functional as F
    from oss.sr.temporal import make_first_frame_prev_hr, warp_prev_hr

    psnr_g: list[float] = []
    psnr_p: list[float] = []
    psnr_a: list[float] = []
    psnr_bic: list[float] = []
    lpips_g: list[float] = []
    lpips_p: list[float] = []
    lpips_a: list[float] = []
    lpips_bic: list[float] = []
    tstab_g: list[float] = []
    tstab_p: list[float] = []
    tstab_a: list[float] = []

    def _lpips(pred, target) -> float | None:
        if lpips_fn is None:
            return None
        p = pred.unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        t = target.unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        return float(lpips_fn(p, t).item())

    pixel_scale = model_pixel.scale
    gauss_scale = engine_gaussian.model.scale

    with torch.no_grad():
        for batch in loader:
            if len(psnr_g) >= n_samples_remaining:
                break

            t_lr = batch["t_lr"].to(device)
            t_depth = batch["t_depth"].to(device)
            t_motion = batch["t_motion"].to(device)
            t_normals = batch["t_normals"].to(device)
            t_canvas = batch["t_canvas"].to(device)

            p_lr = batch["tp1_lr"].to(device)
            p_depth = batch["tp1_depth"].to(device)
            p_motion = batch["tp1_motion"].to(device)
            p_normals = batch["tp1_normals"].to(device)
            p_canvas = batch["tp1_canvas"].to(device)
            p_gt = batch["tp1_gt_hr"].to(device)

            H_hr, W_hr = p_gt.shape[-2:]

            # --- Baseline (single-frame v4) at t and t+1 -----------------
            x_t = _make_12ch(t_lr, t_depth, t_motion, t_normals, t_canvas)
            base_out_t = model_baseline(x_t).clamp(0.0, 1.0)

            x_tp1 = _make_12ch(p_lr, p_depth, p_motion, p_normals, p_canvas)
            base_out_tp1 = model_baseline(x_tp1).clamp(0.0, 1.0)

            # --- Pixel-temporal (v5-P) -----------------------------------
            depth_hr_t = F.interpolate(
                t_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
            )
            depth_hr_tp1 = F.interpolate(
                p_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
            )
            prev_hr_t = make_first_frame_prev_hr(t_lr, scale=pixel_scale)
            pix_out_t = model_pixel(
                lr_inputs=x_t, prev_hr=prev_hr_t,
                depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t,
                motion_lr=t_motion,
            ).clamp(0.0, 1.0)
            # Cold-start regime at t+1: prev_hr = baseline_output_at_t.
            # Motion fed in is t_motion (forward flow t->t+1 lives at frame t),
            # NOT p_motion. Same convention as the pixel held-out fix at 38cf507.
            pix_out_tp1 = model_pixel(
                lr_inputs=x_tp1, prev_hr=base_out_t.detach(),
                depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                motion_lr=t_motion,
            ).clamp(0.0, 1.0)

            # --- Gaussian-temporal (v5-G) — stateful engine, B=1 ---------
            # The engine carries a GaussianField across calls. Each held-out
            # pair is independent, so we reset the engine before frame t,
            # feed t (seeds the field via first-frame densification), then
            # feed t+1 with t_motion (forward flow t->t+1) — the engine
            # warps the prior field by motion_lr internally.
            B = t_lr.shape[0]
            gauss_out_t = torch.zeros_like(p_gt)
            gauss_out_tp1 = torch.zeros_like(p_gt)
            for bi in range(B):
                engine_gaussian.reset()
                # frame t — seeds field
                xt_b = x_t[bi : bi + 1]
                mt_b = t_motion[bi : bi + 1]
                out_t_b = engine_gaussian(lr_inputs=xt_b, motion_lr=mt_b).clamp(0.0, 1.0)
                gauss_out_t[bi] = out_t_b[0]
                # frame t+1 — scored output. Motion fed is t_motion (the
                # warp from t -> t+1 lives at frame t), matching the same
                # flow-direction convention as the pixel held-out script.
                xp_b = x_tp1[bi : bi + 1]
                out_tp1_b = engine_gaussian(lr_inputs=xp_b, motion_lr=mt_b).clamp(0.0, 1.0)
                gauss_out_tp1[bi] = out_tp1_b[0]

            # --- Bicubic upsample of LR_{t+1} -----------------------------
            bic_tp1 = F.interpolate(
                p_lr, size=(H_hr, W_hr), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)

            # --- Temporal stability ---------------------------------------
            # |warp(out_t, motion_t->t+1) - out_{t+1}|_1, mean per sample.
            # t_motion is the forward flow t->t+1 (lives at frame t).
            warped_g = warp_prev_hr(gauss_out_t, t_motion, scale=gauss_scale)
            warped_p = warp_prev_hr(pix_out_t,  t_motion, scale=pixel_scale)
            warped_a = warp_prev_hr(base_out_t, t_motion, scale=pixel_scale)

            for b_idx in range(p_lr.shape[0]):
                if len(psnr_g) >= n_samples_remaining:
                    break
                psnr_g.append(_psnr(gauss_out_tp1[b_idx], p_gt[b_idx]))
                psnr_p.append(_psnr(pix_out_tp1[b_idx], p_gt[b_idx]))
                psnr_a.append(_psnr(base_out_tp1[b_idx], p_gt[b_idx]))
                psnr_bic.append(_psnr(bic_tp1[b_idx], p_gt[b_idx]))

                lg = _lpips(gauss_out_tp1[b_idx], p_gt[b_idx])
                lp = _lpips(pix_out_tp1[b_idx], p_gt[b_idx])
                la = _lpips(base_out_tp1[b_idx], p_gt[b_idx])
                lc = _lpips(bic_tp1[b_idx], p_gt[b_idx])
                if lg is not None:
                    lpips_g.append(lg)
                    lpips_p.append(lp)
                    lpips_a.append(la)
                    lpips_bic.append(lc)

                tstab_g.append(float(
                    (warped_g[b_idx] - gauss_out_tp1[b_idx]).abs().mean().item()
                ))
                tstab_p.append(float(
                    (warped_p[b_idx] - pix_out_tp1[b_idx]).abs().mean().item()
                ))
                tstab_a.append(float(
                    (warped_a[b_idx] - base_out_tp1[b_idx]).abs().mean().item()
                ))

    return {
        "psnr_gaussian": psnr_g,
        "psnr_pixel": psnr_p,
        "psnr_baseline": psnr_a,
        "psnr_bicubic": psnr_bic,
        "lpips_gaussian": lpips_g,
        "lpips_pixel": lpips_p,
        "lpips_baseline": lpips_a,
        "lpips_bicubic": lpips_bic,
        "tstab_gaussian": tstab_g,
        "tstab_pixel": tstab_p,
        "tstab_baseline": tstab_a,
    }


def _merge_results(
    *results: dict[str, list[float]]
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.items():
            merged.setdefault(k, []).extend(v)
    return merged


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_device() -> str:
    """Best-effort cuda/cpu default; falls back to ``"cpu"`` if torch is absent.

    Keeps ``--help`` working in environments where torch isn't installed
    (the Task 12 verification gate runs on a vanilla interpreter).
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-gaussian", type=Path, required=True,
                   help="v5-gaussian-temporal checkpoint (.pt). "
                        "held_out_results.json is written next to it.")
    p.add_argument("--ckpt-pixel", type=Path, required=True,
                   help="v5-pixel-temporal checkpoint (.pt).")
    p.add_argument("--ckpt-baseline", type=Path, required=True,
                   help="v4 single-frame baseline checkpoint (.pt).")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="Held-out TartanAir trajectory root.")
    p.add_argument("--sintel-root", type=Path, default=None,
                   help="Held-out Sintel root (uses 'clean' pass).")
    p.add_argument("--device", type=str, default=_default_device(),
                   help="cuda if available else cpu")
    p.add_argument("--n-samples", type=int, default=64,
                   help="Total held-out frames to evaluate across both datasets.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Defer torch import to inside main() so ``--help`` works without torch.
    import torch
    device = args.device
    torch.manual_seed(args.seed)

    if args.tartanair_root is None and args.sintel_root is None:
        print("FAIL: provide at least one of --tartanair-root / --sintel-root")
        return 1

    print(f"Loading gaussian: {args.ckpt_gaussian}")
    engine_gaussian = _load_gaussian_engine(args.ckpt_gaussian, device)
    print(f"Loading pixel:    {args.ckpt_pixel}")
    model_pixel = _load_pixel_temporal(args.ckpt_pixel, device)
    print(f"Loading baseline: {args.ckpt_baseline}")
    model_baseline = _load_baseline(args.ckpt_baseline, device)

    # LPIPS — single instance shared across all four reconstructions.
    lpips_fn = None
    try:
        import lpips  # type: ignore[import-not-found]
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.train(False)
    except Exception as e:
        print(f"WARN: LPIPS unavailable ({e}) - PSNR + temporal stability only")

    # Build deterministic loaders.
    loaders: list[tuple[str, "DataLoader"]] = []
    if args.tartanair_root is not None:
        loader = _build_pair_loader(
            "tartanair", args.tartanair_root, args.batch_size,
        )
        if loader is not None:
            loaders.append(("tartanair", loader))
            print(f"tartanair held-out pairs: {len(loader.dataset)}")
    if args.sintel_root is not None:
        loader = _build_pair_loader(
            "sintel", args.sintel_root, args.batch_size,
        )
        if loader is not None:
            loaders.append(("sintel", loader))
            print(f"sintel held-out pairs: {len(loader.dataset)}")

    if not loaders:
        print("FAIL: neither dataset produced any sequential pairs")
        return 1

    # Evenly split target sample budget across loaders (round up).
    per_loader = max(1, math.ceil(args.n_samples / len(loaders)))
    per_dataset_results: dict[str, dict[str, list[float]]] = {}
    for name, loader in loaders:
        print(f"-- evaluating {name} (target ~{per_loader} samples) --")
        per_dataset_results[name] = _eval_loader(
            loader,
            engine_gaussian=engine_gaussian,
            model_pixel=model_pixel,
            model_baseline=model_baseline,
            lpips_fn=lpips_fn,
            n_samples_remaining=per_loader,
            device=device,
        )

    merged = _merge_results(*per_dataset_results.values())
    n = len(merged["psnr_gaussian"])
    if n == 0:
        print("FAIL: no samples evaluated")
        return 1

    # ---- Print results ----
    psnr_g = merged["psnr_gaussian"]
    psnr_p = merged["psnr_pixel"]
    psnr_a = merged["psnr_baseline"]
    psnr_c = merged["psnr_bicubic"]
    lpips_g = merged["lpips_gaussian"]
    lpips_p = merged["lpips_pixel"]
    lpips_a = merged["lpips_baseline"]
    lpips_c = merged["lpips_bicubic"]
    tstab_g = merged["tstab_gaussian"]
    tstab_p = merged["tstab_pixel"]
    tstab_a = merged["tstab_baseline"]

    print()
    print(f"=== held-out fixed-batch eval (n={n}) ===")
    print(f"  ckpt_gaussian (G) = {args.ckpt_gaussian.name}")
    print(f"  ckpt_pixel    (P) = {args.ckpt_pixel.name}")
    print(f"  ckpt_baseline (A) = {args.ckpt_baseline.name}")
    print()
    print(f"PSNR (dB, higher is better)")
    print(f"  A (baseline) : {_mean(psnr_a):6.3f}")
    print(f"  P (pixel)    : {_mean(psnr_p):6.3f}")
    print(f"  G (gaussian) : {_mean(psnr_g):6.3f}")
    print(f"  bicubic      : {_mean(psnr_c):6.3f}")
    print(f"  G-vs-P       : {_mean(psnr_g)-_mean(psnr_p):+6.3f} dB")
    print(f"  G-vs-A       : {_mean(psnr_g)-_mean(psnr_a):+6.3f} dB")
    print(f"  P-vs-A       : {_mean(psnr_p)-_mean(psnr_a):+6.3f} dB")
    # Per-sample win counts (race rule: strict > ; tie != G win).
    g_gt_p = sum(1 for g, q in zip(psnr_g, psnr_p) if g > q)
    g_gt_a = sum(1 for g, a in zip(psnr_g, psnr_a) if g > a)
    p_gt_a = sum(1 for q, a in zip(psnr_p, psnr_a) if q > a)
    print(f"  G>P (strict) : {g_gt_p}/{n}   (race rule: tie != G win)")
    print(f"  G>A          : {g_gt_a}/{n}")
    print(f"  P>A          : {p_gt_a}/{n}")
    print(f"  G>bicubic    : {sum(1 for g,c in zip(psnr_g,psnr_c) if g>c)}/{n}")
    print(f"  P>bicubic    : {sum(1 for q,c in zip(psnr_p,psnr_c) if q>c)}/{n}")
    print(f"  A>bicubic    : {sum(1 for a,c in zip(psnr_a,psnr_c) if a>c)}/{n}")

    g_lt_p_lpips: int | None = None
    g_lt_a_lpips: int | None = None
    p_lt_a_lpips: int | None = None
    if lpips_a:
        a_mean = _mean(lpips_a)
        p_mean = _mean(lpips_p)
        g_mean = _mean(lpips_g)
        rel_gp = 100.0 * (g_mean - p_mean) / p_mean if p_mean else 0.0
        rel_ga = 100.0 * (g_mean - a_mean) / a_mean if a_mean else 0.0
        print()
        print(f"LPIPS-VGG (lower is better)")
        print(f"  A (baseline) : {a_mean:6.4f}")
        print(f"  P (pixel)    : {p_mean:6.4f}")
        print(f"  G (gaussian) : {g_mean:6.4f}")
        print(f"  bicubic      : {_mean(lpips_c):6.4f}")
        print(f"  G-vs-P       : {g_mean - p_mean:+7.4f}  ({rel_gp:+5.1f}%)")
        print(f"  G-vs-A       : {g_mean - a_mean:+7.4f}  ({rel_ga:+5.1f}%)")
        # Race rule: G must be STRICTLY less than P (lower LPIPS = better).
        g_lt_p_lpips = sum(1 for g, q in zip(lpips_g, lpips_p) if g < q)
        g_lt_a_lpips = sum(1 for g, a in zip(lpips_g, lpips_a) if g < a)
        p_lt_a_lpips = sum(1 for q, a in zip(lpips_p, lpips_a) if q < a)
        print(f"  G<P (strict) : {g_lt_p_lpips}/{n}   (race rule: tie != G win)")
        print(f"  G<A          : {g_lt_a_lpips}/{n}")
        print(f"  P<A          : {p_lt_a_lpips}/{n}")
        print(f"  G<bicubic    : {sum(1 for g,c in zip(lpips_g,lpips_c) if g<c)}/{n}")
        print(f"  P<bicubic    : {sum(1 for q,c in zip(lpips_p,lpips_c) if q<c)}/{n}")

    print()
    print(f"Temporal stability (mean(|warp(out_t, motion_t->t+1) - out_t+1|_1), lower is better)")
    print(f"  A (baseline) : {_mean(tstab_a):7.5f}")
    print(f"  P (pixel)    : {_mean(tstab_p):7.5f}")
    print(f"  G (gaussian) : {_mean(tstab_g):7.5f}")
    if _mean(tstab_a) > 0:
        gp_ratio = _mean(tstab_g) / _mean(tstab_p) if _mean(tstab_p) > 0 else float("nan")
        ga_ratio = _mean(tstab_g) / _mean(tstab_a)
        pa_ratio = _mean(tstab_p) / _mean(tstab_a)
        print(f"  G/P ratio    : {gp_ratio:5.3f}  (race rule: G must be < P, i.e. ratio < 1.0)")
        print(f"  G/A ratio    : {ga_ratio:5.3f}  (spec target: <= 0.5)")
        print(f"  P/A ratio    : {pa_ratio:5.3f}")
    print()

    # Joint criterion (PSNR > bicubic AND LPIPS < bicubic) for G and P.
    joint_g = 0
    joint_p = 0
    if lpips_a:
        for pg, pp, pc, lg, lp, lc in zip(
            psnr_g, psnr_p, psnr_c, lpips_g, lpips_p, lpips_c
        ):
            if pg > pc and lg < lc:
                joint_g += 1
            if pp > pc and lp < lc:
                joint_p += 1
        pct_g = 100.0 * joint_g / n
        pct_p = 100.0 * joint_p / n
        print(
            f"  G beats bicubic on PSNR AND LPIPS : "
            f"{joint_g}/{n}  ({pct_g:.1f}%)  (spec target: >= 95%)"
        )
        print(
            f"  P beats bicubic on PSNR AND LPIPS : "
            f"{joint_p}/{n}  ({pct_p:.1f}%)  (spec target: >= 95%)"
        )
        print()

    # ---- Write JSON next to the gaussian checkpoint ----
    out: dict[str, Any] = {
        "n_samples": n,
        "ckpt_gaussian": str(args.ckpt_gaussian),
        "ckpt_pixel": str(args.ckpt_pixel),
        "ckpt_baseline": str(args.ckpt_baseline),
        "datasets": {name: len(r["psnr_gaussian"]) for name, r in per_dataset_results.items()},
        "psnr": {
            "baseline_mean": _mean(psnr_a),
            "pixel_mean": _mean(psnr_p),
            "gaussian_mean": _mean(psnr_g),
            "bicubic_mean": _mean(psnr_c),
            "delta_g_minus_p": _mean(psnr_g) - _mean(psnr_p),
            "delta_g_minus_a": _mean(psnr_g) - _mean(psnr_a),
            "delta_p_minus_a": _mean(psnr_p) - _mean(psnr_a),
            "G_gt_P": g_gt_p,
            "G_gt_A": g_gt_a,
            "P_gt_A": p_gt_a,
            "G_gt_bicubic": sum(1 for g, c in zip(psnr_g, psnr_c) if g > c),
            "P_gt_bicubic": sum(1 for q, c in zip(psnr_p, psnr_c) if q > c),
            "A_gt_bicubic": sum(1 for a, c in zip(psnr_a, psnr_c) if a > c),
        },
        "lpips": {
            "baseline_mean": _mean(lpips_a) if lpips_a else None,
            "pixel_mean": _mean(lpips_p) if lpips_a else None,
            "gaussian_mean": _mean(lpips_g) if lpips_a else None,
            "bicubic_mean": _mean(lpips_c) if lpips_a else None,
            "G_lt_P": g_lt_p_lpips,
            "G_lt_A": g_lt_a_lpips,
            "P_lt_A": p_lt_a_lpips,
            "G_lt_bicubic": (
                sum(1 for g, c in zip(lpips_g, lpips_c) if g < c) if lpips_a else None
            ),
            "P_lt_bicubic": (
                sum(1 for q, c in zip(lpips_p, lpips_c) if q < c) if lpips_a else None
            ),
        },
        "temporal_stability": {
            "baseline_mean": _mean(tstab_a),
            "pixel_mean": _mean(tstab_p),
            "gaussian_mean": _mean(tstab_g),
            "ratio_g_over_p": (
                (_mean(tstab_g) / _mean(tstab_p)) if _mean(tstab_p) > 0 else None
            ),
            "ratio_g_over_a": (
                (_mean(tstab_g) / _mean(tstab_a)) if _mean(tstab_a) > 0 else None
            ),
            "ratio_p_over_a": (
                (_mean(tstab_p) / _mean(tstab_a)) if _mean(tstab_a) > 0 else None
            ),
        },
        "joint_g_beats_bicubic": joint_g if lpips_a else None,
        "joint_g_beats_bicubic_pct": (100.0 * joint_g / n) if lpips_a and n else None,
        "joint_p_beats_bicubic": joint_p if lpips_a else None,
        "joint_p_beats_bicubic_pct": (100.0 * joint_p / n) if lpips_a and n else None,
        "race_rule": (
            "Gaussian must explicitly beat pixel; tie != Gaussian win. "
            "Strict > on PSNR, strict < on LPIPS, strict < on temporal-stability."
        ),
    }
    json_path = args.ckpt_gaussian.parent / "held_out_results.json"
    with json_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

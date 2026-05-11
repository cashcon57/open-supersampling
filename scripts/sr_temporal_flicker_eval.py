#!/usr/bin/env python3
"""Continuous-trajectory flicker eval for OSS v6.x temporal SR models.

The existing held-out eval (``scripts/sr_temporal_held_out.py``) measures
temporal stability pair-wise: each held-out pair ``(t, t+1)`` resets the
canvas, runs both frames, and reports ``|warp(out_t, MV) - out_{t+1}|_1``.
That's a good "does the next frame look like the warped previous" signal
but it doesn't capture the harder question: **when the canvas persists
across many consecutive frames, does the output flicker?**

Flicker = per-pixel temporal variance at locations the camera/scene
didn't move. If a wall pixel stays "the same wall" for 30 consecutive
frames but the model paints it 30 slightly different textures, that's
flicker -- visually painful at game frame rates, invisible to LPIPS,
and only marginally visible to pair-wise tstab.

This script:

  1. Loads ``N`` contiguous frames from TartanAir (single sub-trajectory).
  2. Resets the model's canvas state once at the start, then runs the
     model frame-by-frame WITHOUT resetting -- canvas state accumulates
     and warps between frames per ``motion_lr``.
  3. Records the model output for each frame.
  4. Computes:
       a. ``tstab_pairwise[t] = |warp(out_t, MV_t->t+1) - out_{t+1}|.mean()``
          (the same metric as the existing pair-wise eval, but over a
          contiguous trajectory rather than independent pair samples)
       b. ``stable_pixel_variance = var(out[:, mask].std(dim=0))``
          where ``mask`` is the set of pixels whose motion magnitude is
          below a small threshold across the full window (= "this pixel
          is approximately the same patch every frame, so any output
          variance is flicker, not motion").
       c. ``stable_pixel_mean_delta = mean(|out_t - out_{t+1}|)`` at
          low-motion pixels only.

Output: one JSON record per evaluated ckpt. Designed to be cheap enough
to run on a handful of ckpts manually; not part of the live supervisor
loop.

Usage:

    python scripts/sr_temporal_flicker_eval.py \\
        --ckpt-temporal E:/checkpoints/srcnn-v6.2-pico-002/step-00050000.pt \\
        --tartanair-root E:/datasets/tartanair_extracted \\
        --tartanair-env oldtown \\
        --tartanair-difficulty Easy \\
        --tartanair-path P000 \\
        --n-frames 32 \\
        --low-motion-threshold 0.5 \\
        --output /tmp/pico-002-step-50k-flicker.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-temporal", type=Path, required=True,
                   help="v6.x generator checkpoint (.pt).")
    p.add_argument("--tartanair-root", type=Path, required=True,
                   help="TartanAir root (the dir that contains env subdirs).")
    p.add_argument("--tartanair-env", type=str, default="oldtown",
                   help="TartanAir env name (default oldtown).")
    p.add_argument("--tartanair-difficulty", type=str, default="Easy",
                   choices=("Easy", "Hard"),
                   help="TartanAir difficulty band.")
    p.add_argument("--tartanair-path", type=str, default="P000",
                   help="TartanAir sub-trajectory id (e.g. P000).")
    p.add_argument("--n-frames", type=int, default=32,
                   help="Number of contiguous frames to roll. Must be >= 4.")
    p.add_argument("--low-motion-threshold", type=float, default=0.5,
                   help="Per-pixel motion-magnitude threshold (LR pixels) "
                        "below which a pixel is considered low-motion. The "
                        "flicker metrics aggregate over the union of pixels "
                        "low-motion in every frame transition.")
    p.add_argument("--device", type=str, default=_default_device())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path. Without this we print to stdout.")
    return p.parse_args(argv)


def _load_v6_model(ckpt_path: Path, device: str):
    """Mirror the v6 loader from sr_temporal_held_out.py so v6.2-trained
    ckpts (concat fusion + disocclusion spawner + latent rank) instantiate
    correctly."""
    import torch
    from oss.sr.v6.model import V6Config, V6Model
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {}) if isinstance(ck, dict) else {}
    cfg_data = ck.get("v6_config", {}) if isinstance(ck, dict) else {}
    if not isinstance(cfg_data, dict):
        cfg_data = {}
    cfg_kwargs = dict(cfg_data)
    cfg_kwargs.setdefault("backbone", saved.get("backbone", "hat-tiny"))
    cfg_kwargs.setdefault("in_channels", int(saved.get("in_channels", 9)))
    cfg_kwargs.setdefault("scale", int(saved.get("scale", 2)))
    cfg_kwargs.setdefault("color_activation", saved.get("color_activation", "hdr"))
    cfg_kwargs.setdefault("spawn_offset_random", bool(saved.get("spawn_offset_random", False)))
    cfg_kwargs.setdefault("rasterizer_overlap", int(saved.get("rasterizer_overlap", 0)))
    if "fusion_mode" in saved:
        cfg_kwargs.setdefault("fusion_mode", str(saved["fusion_mode"]))
    if "spawner_mode" in saved:
        cfg_kwargs.setdefault("spawner_mode", str(saved["spawner_mode"]))
    if "latent_rank" in saved:
        cfg_kwargs.setdefault("latent_rank", int(saved["latent_rank"]))
    model = V6Model(V6Config(**cfg_kwargs)).to(device)
    state = None
    for key in ("v6_model", "model", "model_state_dict", "generator", "state_dict"):
        if key in ck:
            state = ck[key]
            break
    if state is None:
        raise KeyError(f"checkpoint {ckpt_path} has no v6 state dict")
    model.load_state_dict(state, strict=False)
    model.train(False)
    return model


def _build_continuous_trajectory(
    root: Path, env: str, difficulty: str, path_id: str, n_frames: int,
) -> list[dict[str, Any]]:
    """Load n_frames contiguous frames from a single TartanAir sub-trajectory.

    Returns a list of dicts, one per frame, with tensors already loaded:
    lr, gt_hr, depth, motion, normals. Uses the same TartanAirGaussianDataset
    that the existing eval uses so degradation/synth is consistent.
    """
    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.sr.temporal import adapt_tartanair
    ds = adapt_tartanair(TartanAirGaussianDataset(root=root, scale=2.0))
    # adapt_tartanair gives us per-sample dicts. Subset to the requested
    # sub-trajectory and take the first n_frames samples in index order.
    selected: list[dict[str, Any]] = []
    target_prefix = (env, difficulty, path_id)
    for idx in range(len(ds)):
        try:
            sample = ds[idx]
        except Exception:
            continue
        # The dataset stores trajectory metadata; the exact key depends on
        # the adapter. We match by inspecting available identifying fields.
        meta = sample.get("meta") if isinstance(sample, dict) else None
        if meta is None:
            # Fall back to filename match against the requested path.
            path_hint = sample.get("source_path", "") if isinstance(sample, dict) else ""
            if not all(p in str(path_hint) for p in target_prefix):
                continue
        else:
            if not (meta.get("env") == env and meta.get("difficulty") == difficulty
                    and meta.get("path") == path_id):
                continue
        selected.append(sample)
        if len(selected) >= n_frames:
            break
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    import torch.nn.functional as F

    if args.n_frames < 4:
        print(f"FAIL: --n-frames must be >= 4 (got {args.n_frames})")
        return 1

    torch.manual_seed(args.seed)
    device = args.device

    print(f"Loading model: {args.ckpt_temporal}")
    model = _load_v6_model(args.ckpt_temporal, device)

    print(f"Loading {args.n_frames} frames from "
          f"{args.tartanair_env}/{args.tartanair_difficulty}/{args.tartanair_path}")
    frames = _build_continuous_trajectory(
        root=args.tartanair_root,
        env=args.tartanair_env,
        difficulty=args.tartanair_difficulty,
        path_id=args.tartanair_path,
        n_frames=args.n_frames,
    )
    if len(frames) < args.n_frames:
        print(f"WARN: only {len(frames)} contiguous frames found (asked for {args.n_frames})")
    if len(frames) < 4:
        print(f"FAIL: not enough frames to compute flicker (got {len(frames)})")
        return 1

    # Reset canvas state once at the start; do NOT reset between frames.
    if hasattr(model, "reset_state"):
        model.reset_state(device=torch.device(device))

    outputs: list[torch.Tensor] = []
    motions: list[torch.Tensor] = []
    with torch.inference_mode():
        for frame_idx, sample in enumerate(frames):
            lr = sample["lr"].to(device).unsqueeze(0) if sample["lr"].dim() == 3 else sample["lr"].to(device)
            depth = sample["depth"].to(device).unsqueeze(0) if sample["depth"].dim() == 3 else sample["depth"].to(device)
            motion = sample["motion"].to(device).unsqueeze(0) if sample["motion"].dim() == 3 else sample["motion"].to(device)
            normals = sample["normals"].to(device).unsqueeze(0) if sample["normals"].dim() == 3 else sample["normals"].to(device)
            # v6 expects 9 input channels: lr(3) + depth(1) + motion(2) + normals(3).
            lr_in = torch.cat([lr, depth, motion, normals], dim=1)
            scale = int(getattr(model.cfg, "scale", 2))
            depth_hr = F.interpolate(depth, scale_factor=scale, mode="bilinear", align_corners=False)
            prev_depth_hr = depth_hr if not outputs else F.interpolate(motions[-1], scale_factor=scale, mode="bilinear", align_corners=False)
            out = model(
                lr_inputs=lr_in,
                motion_lr=motion if outputs else None,
                depth_hr_curr=depth_hr,
                depth_hr_prev=depth_hr if not outputs else prev_depth_hr,
                frame_index=frame_idx,
            ).clamp(0.0, 1.0)
            outputs.append(out.squeeze(0).cpu())
            motions.append(motion.cpu())

    # Stack outputs: (T, 3, H, W)
    out_stack = torch.stack(outputs, dim=0)
    motion_stack = torch.stack([m.squeeze(0) for m in motions], dim=0)  # (T, 2, H, W) at LR
    T = out_stack.shape[0]
    H_hr, W_hr = out_stack.shape[-2:]

    # Pair-wise tstab (warp prev to next via motion at t)
    from oss.sr.temporal import warp_prev_hr
    pairwise: list[float] = []
    for t in range(T - 1):
        warped = warp_prev_hr(out_stack[t:t+1].to(device), motion_stack[t:t+1].to(device), scale=2)
        delta = (warped.squeeze(0).cpu() - out_stack[t + 1]).abs().mean().item()
        pairwise.append(float(delta))

    # Low-motion mask: a pixel is low-motion if its motion magnitude (LR)
    # stays below threshold across ALL transitions. Upsample to HR.
    motion_mag_lr = motion_stack.norm(dim=1)  # (T, H_lr, W_lr)
    low_motion_lr = (motion_mag_lr < args.low_motion_threshold).all(dim=0)  # (H_lr, W_lr)
    low_motion_hr = F.interpolate(
        low_motion_lr.float().unsqueeze(0).unsqueeze(0),
        size=(H_hr, W_hr), mode="nearest"
    ).squeeze(0).squeeze(0).bool()  # (H_hr, W_hr)
    n_stable = int(low_motion_hr.sum().item())
    n_total = int(low_motion_hr.numel())
    stable_frac = n_stable / max(1, n_total)

    flicker_var = float("nan")
    flicker_mean_delta = float("nan")
    if n_stable > 0:
        masked = out_stack[:, :, low_motion_hr]  # (T, 3, n_stable)
        # Per-stable-pixel temporal variance across the T window, mean over channels + pixels.
        temporal_var = masked.var(dim=0, unbiased=False).mean().item()
        flicker_var = float(temporal_var)
        # Frame-to-frame absolute delta on the low-motion mask.
        deltas = (out_stack[1:, :, low_motion_hr] - out_stack[:-1, :, low_motion_hr]).abs()
        flicker_mean_delta = float(deltas.mean().item())

    report = {
        "ckpt": str(args.ckpt_temporal),
        "step": int(args.ckpt_temporal.stem.split("-")[-1]) if "-" in args.ckpt_temporal.stem else None,
        "trajectory": {
            "env": args.tartanair_env,
            "difficulty": args.tartanair_difficulty,
            "path": args.tartanair_path,
        },
        "n_frames_evaluated": T,
        "low_motion_threshold_lr_px": args.low_motion_threshold,
        "stable_pixel_fraction": stable_frac,
        "pairwise_tstab": {
            "mean": sum(pairwise) / max(1, len(pairwise)),
            "max": max(pairwise) if pairwise else float("nan"),
            "per_frame": pairwise,
        },
        "flicker_at_stable_pixels": {
            "temporal_variance": flicker_var,
            "frame_to_frame_mean_delta": flicker_mean_delta,
            "n_stable_pixels": n_stable,
            "n_total_pixels": n_total,
        },
    }

    out_str = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_str + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(out_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

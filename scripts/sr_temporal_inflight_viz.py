#!/usr/bin/env python
"""In-flight visualization for v5-pixel-temporal training.

Watches a checkpoint dir, loads the latest ``step-XXXXX.pt`` periodically,
renders a fixed set of held-out frames as a 4-up comparison strip:

    [ LR (bilinear-up) | bicubic | model | GT ]

Writes ``output_dir/viz/step-XXXXX.png`` after each iteration. Designed to
run as a background loop alongside training; uses CPU inference to avoid
GPU contention with the live training process.

Pair selection is read from the deterministic held-out manifest produced by
``scripts/sr_freeze_held_out_manifest.py``. Default 4 pairs (a small subset
of the full 64 — keeps each iteration under ~30 s on CPU).

Usage::

    python scripts/sr_temporal_inflight_viz.py \\
        --output-dir <train-host-data>/checkpoints/srcnn-v5-pixel-temporal \\
        --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json \\
        --tartanair-root <train-host-data>/datasets/tartanair_extracted \\
        --interval 300 \\
        --n-pairs 4

A companion static file server (``python -m http.server``) can serve the viz
dir to a browser; see launch-status notes for the actual orphan-spawn
command.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``python scripts/...`` from a system Python without installing the
# package. Mirrors the other v5-pixel-temporal scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Training output dir containing step-*.pt checkpoints.")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Held-out manifest JSON (from sr_freeze_held_out_manifest.py).")
    p.add_argument("--tartanair-root", type=Path, default=None,
                   help="TartanAir root for resolving manifest pair paths.")
    p.add_argument("--n-pairs", type=int, default=4,
                   help="Number of pairs from the manifest to render per iteration.")
    p.add_argument("--interval", type=int, default=300,
                   help="Seconds between viz iterations (default 300 = 5 min).")
    p.add_argument("--device", default="cpu",
                   help="Inference device. Default cpu (avoids contention with training GPU).")
    p.add_argument("--once", action="store_true",
                   help="Render one iteration and exit (smoke / one-shot).")
    return p.parse_args(argv)


def _latest_ckpt(output_dir: Path) -> Path | None:
    ckpts = sorted(output_dir.glob("step-*.pt"))
    return ckpts[-1] if ckpts else None


def _render_iteration(
    *,
    ckpt_path: Path,
    manifest_path: Path,
    tartanair_root: Path,
    output_dir: Path,
    n_pairs: int,
    device: str,
) -> Path | None:
    """Render a single 4-up comparison strip and write to viz/step-XXXXX.png."""
    import torch
    import torch.nn.functional as F

    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.sr.temporal import (
        SequentialPairDataset, TemporalSRModel,
        adapt_tartanair, make_first_frame_prev_hr,
    )
    from oss.sr.temporal.held_out_manifest import load_manifest

    # Step number from filename (e.g. step-00012000.pt -> 12000)
    step_str = ckpt_path.stem.split("-")[-1]
    try:
        step = int(step_str)
    except ValueError:
        step = -1

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True, parents=True)
    out_path = viz_dir / f"step-{step:08d}.png"
    if out_path.exists():
        return None  # already rendered this step

    # Load model.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {})
    tier = saved.get("tier", "standard")
    backbone_kind = saved.get("backbone_kind", "simple")
    model = TemporalSRModel(in_channels=12, scale=2, tier=tier, backbone_kind=backbone_kind).to(device)
    if "temporal_model" in ck:
        model.load_state_dict(ck["temporal_model"])
    elif "sr_model" in ck:
        model.backbone.load_state_dict(ck["sr_model"])
    model.train(False)

    # Load manifest + dataset. The dataset's __getitem__ already produces an
    # LR frame box-downsampled from HR, which is enough for an in-flight
    # visual sanity check (the engine-aliased LR-synth path used in training
    # is more realistic but not needed for "is the model improving" eyeballing).
    manifest = load_manifest(manifest_path)
    pairs_meta = manifest["pairs"][:n_pairs]
    base = adapt_tartanair(TartanAirGaussianDataset(root=tartanair_root, scale=manifest["lr_scale"]))
    pair_ds = SequentialPairDataset(base)

    # Map manifest pair entries to base-dataset (idx_t, idx_t+1) by trajectory match.
    base_items = list(base._items)
    rendered_strips: list[torch.Tensor] = []
    for pm in pairs_meta:
        traj = pm["trajectory"]
        idx_t = pm["idx_t"]
        # Resolve to a base-dataset row whose trajectory_key matches and whose frame index is idx_t.
        base_idx_t = None
        for i, item in enumerate(base_items):
            if str(item[0].parent.parent) == traj and i % 10000 < 10000:  # cheap scan
                # Frame number is in the filename, e.g., 000123_left.png
                fname = item[0].name
                fnum = int(fname.split("_")[0])
                if fnum == idx_t:
                    base_idx_t = i
                    break
        if base_idx_t is None:
            continue
        ex_t = base[base_idx_t]
        # Use the dataset's already-prepared LR frame (box-downsampled HR).
        lr_t = ex_t.lr_frame
        depth = ex_t.depth.to(device)
        motion = ex_t.motion.to(device)
        normals = (ex_t.normals if ex_t.normals is not None else
                   torch.zeros((3, *lr_t.shape[-2:]), dtype=lr_t.dtype)).to(device)
        canvas = ex_t.canvas_hint.to(device) if ex_t.canvas_hint is not None else torch.zeros(
            (3, *lr_t.shape[-2:]), dtype=lr_t.dtype, device=device)
        lr_t = lr_t.to(device)

        x12 = torch.cat([lr_t, depth, motion, normals, canvas], dim=0).unsqueeze(0)
        H_lr, W_lr = lr_t.shape[-2:]
        H_hr, W_hr = H_lr * model.scale, W_lr * model.scale
        prev_hr = make_first_frame_prev_hr(lr_t.unsqueeze(0)[:, :3], scale=model.scale)
        depth_hr = F.interpolate(depth.unsqueeze(0), size=(H_hr, W_hr), mode="bilinear",
                                 align_corners=False)

        with torch.no_grad():
            model_out = model(
                lr_inputs=x12, prev_hr=prev_hr, depth_hr_curr=depth_hr,
                depth_hr_prev=depth_hr, motion_lr=motion.unsqueeze(0),
            ).clamp(0.0, 1.0)
        bicubic_hr = F.interpolate(lr_t.unsqueeze(0)[:, :3], size=(H_hr, W_hr),
                                   mode="bicubic", antialias=True).clamp(0.0, 1.0)
        lr_up = F.interpolate(lr_t.unsqueeze(0)[:, :3], size=(H_hr, W_hr),
                              mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        gt_hr = ex_t.gt_hr_frame.unsqueeze(0).to(device).clamp(0.0, 1.0)

        # Stack horizontally: [LR-up | bicubic | model | GT]
        strip = torch.cat([lr_up[0], bicubic_hr[0], model_out[0], gt_hr[0]], dim=-1)
        rendered_strips.append(strip.cpu())
    if not rendered_strips:
        return None

    # Stack vertically across the n_pairs strips.
    composite = torch.cat(rendered_strips, dim=-2)
    # Save as PNG.
    from torchvision.utils import save_image
    save_image(composite, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.output_dir.is_dir():
        print(f"output-dir {args.output_dir} does not exist", file=sys.stderr)
        return 1
    if not args.manifest.exists():
        print(f"manifest {args.manifest} does not exist", file=sys.stderr)
        return 1
    if args.tartanair_root is None or not args.tartanair_root.is_dir():
        print(f"--tartanair-root must point to an existing dir", file=sys.stderr)
        return 1

    print(f"in-flight viz: output_dir={args.output_dir} interval={args.interval}s "
          f"n_pairs={args.n_pairs} device={args.device}", flush=True)

    last_step = -1
    iters = 0
    while True:
        ckpt = _latest_ckpt(args.output_dir)
        if ckpt is None:
            print(f"  no checkpoints yet at {args.output_dir}", flush=True)
        else:
            try:
                step = int(ckpt.stem.split("-")[-1])
            except ValueError:
                step = -1
            if step != last_step:
                t0 = time.monotonic()
                try:
                    out = _render_iteration(
                        ckpt_path=ckpt, manifest_path=args.manifest,
                        tartanair_root=args.tartanair_root, output_dir=args.output_dir,
                        n_pairs=args.n_pairs, device=args.device,
                    )
                    elapsed = time.monotonic() - t0
                    if out is None:
                        print(f"  step {step}: no new viz", flush=True)
                    else:
                        print(f"  step {step}: rendered {out} in {elapsed:.1f}s", flush=True)
                    last_step = step
                except Exception as e:
                    print(f"  step {step}: render failed: {e}", flush=True)
            else:
                print(f"  step {step}: unchanged, skipping", flush=True)
        iters += 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

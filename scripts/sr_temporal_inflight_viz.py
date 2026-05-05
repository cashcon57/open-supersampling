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
    p.add_argument("--ckpt-baseline", type=Path, default=None,
                   help="Optional v4-baseline ckpt path; if provided, viz adds "
                        "a v4-baseline column for direct A/B with v5-temporal.")
    p.add_argument("--err-scale", type=float, default=0.2,
                   help="Error heatmap normalization (per-channel L1 absolute "
                        "error mapped to [0, err_scale] -> red colormap).")
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
    ckpt_baseline: Path | None = None,
    err_scale: float = 0.2,
) -> Path | None:
    """Render a single 6-up comparison strip and write to viz/step-XXXXX.png.

    Strip layout (left to right):
      LR-bilinear | bicubic | v4-baseline | v5-temporal | GT | |error|
    """
    import torch
    import torch.nn.functional as F

    from oss.gaussian.data import EngineAliasedLRSynth, TartanAirGaussianDataset
    from oss.sr import build_sr_model
    from oss.sr.temporal import (
        SequentialPairDataset, TemporalSRModel,
        adapt_tartanair, make_first_frame_prev_hr,
        warp_prev_hr,
    )
    from oss.sr.temporal.held_out_manifest import load_manifest, manifest_to_pairs

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

    # Load v5-temporal model.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("args", {})
    tier = saved.get("tier", "standard")
    backbone_kind = saved.get("backbone_kind", "simple")
    if "zero_gbuffer_into_backbone" in saved:
        zero_flag = bool(saved["zero_gbuffer_into_backbone"])
    else:
        # Legacy ckpts: warm-started runs zeroed G-buffer channels into
        # the backbone; from-scratch runs did not.
        zero_flag = bool(saved.get("warm_start"))
    model = TemporalSRModel(
        in_channels=12, scale=2, tier=tier, backbone_kind=backbone_kind,
        zero_gbuffer_into_backbone=zero_flag,
    ).to(device)
    if "temporal_model" in ck:
        model.load_state_dict(ck["temporal_model"])
    elif "sr_model" in ck:
        model.backbone.load_state_dict(ck["sr_model"])
    model.train(False)

    # Load v4-baseline single-frame model (optional column).
    baseline = None
    if ckpt_baseline is not None and ckpt_baseline.exists():
        bck = torch.load(ckpt_baseline, map_location=device, weights_only=False)
        bsaved = bck.get("args", {})
        b_tier = bsaved.get("tier", "standard")
        b_backbone = bsaved.get("sr_backbone", "simple")
        b_kind = "rrdb" if b_backbone == "rrdb" else "simple"
        baseline = build_sr_model(
            model_kind=b_kind, tier=b_tier, in_channels=12, scale=2,
        ).to(device)
        baseline.load_state_dict(bck["sr_model"])
        baseline.train(False)

    # Load manifest + dataset. Codex R5 review fixed three issues:
    #
    #   (a) Distribution match — the dataset is built with
    #       EngineAliasedLRSynth(...) using the manifest's saved LR-synth
    #       config, so the LR fed to the model matches the held-out script's
    #       LR generation regime (rather than a too-clean box-downsample).
    #
    #   (b) Real temporal eval — the manifest's (idx_t, idx_t_plus_1) pair
    #       is honored. We seed prev_hr by running the model on frame t,
    #       then visualize the OUTPUT on frame t+1 using t_motion. That is
    #       the same regime sr_temporal_held_out.py uses; without it the
    #       viz only shows the first-frame fallback path and never exercises
    #       the temporal head's prev-HR warp.
    #
    #   (c) Use manifest_to_pairs(manifest, base) for hard-fail-on-mismatch
    #       pair resolution rather than a silent skip on missing frames.
    manifest = load_manifest(manifest_path)
    pairs_meta = manifest["pairs"][:n_pairs]
    lr_synth = EngineAliasedLRSynth(
        scale=manifest["lr_scale"], **manifest.get("lr_synth_args", {})
    )
    base = adapt_tartanair(
        TartanAirGaussianDataset(
            root=tartanair_root, scale=manifest["lr_scale"], lr_synth=lr_synth,
        )
    )

    # Resolve each manifest pair to base-dataset indices. ``manifest_to_pairs``
    # raises clearly on a path/frame mismatch; subset to the first N_pairs we
    # actually want to visualize.
    sliced = dict(manifest)
    sliced["pairs"] = pairs_meta
    sliced["n_pairs"] = len(pairs_meta)
    resolved = manifest_to_pairs(sliced, base)

    rendered_strips: list[torch.Tensor] = []
    for (base_idx_t, base_idx_tp1) in resolved:
        ex_t = base[base_idx_t]
        ex_tp1 = base[base_idx_tp1]

        def _to_x12(ex):
            lr = ex.lr_frame.to(device)
            depth_lr = ex.depth.to(device)
            motion_lr = ex.motion.to(device)
            normals = (ex.normals if ex.normals is not None else
                       torch.zeros((3, *lr.shape[-2:]), dtype=lr.dtype)).to(device)
            canvas = (ex.canvas_hint if ex.canvas_hint is not None else
                      torch.zeros((3, *lr.shape[-2:]), dtype=lr.dtype)).to(device)
            return lr, depth_lr, motion_lr, normals, canvas

        lr_t, depth_t_lr, motion_t_lr, normals_t, canvas_t = _to_x12(ex_t)
        lr_tp1, depth_tp1_lr, motion_tp1_lr, normals_tp1, canvas_tp1 = _to_x12(ex_tp1)

        x12_t = torch.cat([lr_t, depth_t_lr, motion_t_lr, normals_t, canvas_t], dim=0).unsqueeze(0)
        x12_tp1 = torch.cat([lr_tp1, depth_tp1_lr, motion_tp1_lr, normals_tp1, canvas_tp1], dim=0).unsqueeze(0)

        H_lr, W_lr = lr_t.shape[-2:]
        H_hr, W_hr = H_lr * model.scale, W_lr * model.scale
        depth_hr_t = F.interpolate(depth_t_lr.unsqueeze(0), size=(H_hr, W_hr),
                                   mode="bilinear", align_corners=False)
        depth_hr_tp1 = F.interpolate(depth_tp1_lr.unsqueeze(0), size=(H_hr, W_hr),
                                     mode="bilinear", align_corners=False)

        # Frame t: cold-start (bilinear-LR-up as prev_hr) — same protocol as
        # sr_temporal_held_out.py uses on the first frame in a pair.
        prev_hr_t = make_first_frame_prev_hr(lr_t.unsqueeze(0)[:, :3], scale=model.scale)
        with torch.no_grad():
            out_t = model(
                lr_inputs=x12_t, prev_hr=prev_hr_t,
                depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t,
                motion_lr=motion_t_lr.unsqueeze(0),
            ).clamp(0.0, 1.0)
        # Frame t+1: prev_hr is the model's frame-t output; motion is t_motion
        # (forward flow t -> t+1, lives at frame t — matches the t_motion
        # convention enforced in sr_temporal_held_out.py post-38cf507).
        with torch.no_grad():
            out_tp1 = model(
                lr_inputs=x12_tp1, prev_hr=out_t.detach(),
                depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                motion_lr=motion_t_lr.unsqueeze(0),
            ).clamp(0.0, 1.0)

        bicubic_tp1 = F.interpolate(lr_tp1.unsqueeze(0)[:, :3], size=(H_hr, W_hr),
                                    mode="bicubic", antialias=True).clamp(0.0, 1.0)
        lr_up_tp1 = F.interpolate(lr_tp1.unsqueeze(0)[:, :3], size=(H_hr, W_hr),
                                  mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        gt_tp1 = ex_tp1.gt_hr_frame.unsqueeze(0).to(device).clamp(0.0, 1.0)

        # v4-baseline column (single-frame; no temporal/prev_hr regime).
        # MUST match the SRGD training distribution v4 was trained against
        # (depth/motion/normals = 0, normals[2]=1.0). Feeding TartanAir's
        # real G-buffers into v4 produces the same chromatic-dispersion
        # garbage that motivated the b2fa647 fix in TemporalSRModel.
        if baseline is not None:
            with torch.no_grad():
                base_in = torch.zeros_like(x12_tp1)
                base_in[:, :3] = x12_tp1[:, :3]
                if base_in.shape[1] >= 7:
                    base_in[:, 6] = 1.0
                base_out_tp1 = baseline(base_in).clamp(0.0, 1.0)
        else:
            base_out_tp1 = bicubic_tp1  # fallback so strip width stays consistent

        # Per-pixel L1 error between v5-temporal and GT, normalized to
        # [0, err_scale] then mapped to a black->red->yellow gradient. Reveals
        # WHERE the model fails (edges? dark regions? high-freq texture?).
        # Channels-collapsed via mean so a single error magnitude per pixel.
        err = (out_tp1[0] - gt_tp1[0]).abs().mean(dim=0, keepdim=True)  # (1, H, W)
        err_norm = (err / max(err_scale, 1e-6)).clamp(0.0, 1.0)
        # Hot-iron gradient: lo (black) -> mid (red) -> hi (yellow).
        # red = clamp(2*x), green = clamp(2*x - 1), blue = 0
        red = (err_norm * 2.0).clamp(0.0, 1.0)
        green = (err_norm * 2.0 - 1.0).clamp(0.0, 1.0)
        blue = torch.zeros_like(err_norm)
        err_rgb = torch.cat([red, green, blue], dim=0)

        # Stack horizontally: [LR-up | bicubic | v4 | v5-temporal | GT | |err|]
        strip = torch.cat([
            lr_up_tp1[0], bicubic_tp1[0], base_out_tp1[0],
            out_tp1[0], gt_tp1[0], err_rgb,
        ], dim=-1)
        rendered_strips.append(strip.cpu())
    if not rendered_strips:
        return None

    # Stack vertically across the n_pairs strips.
    composite = torch.cat(rendered_strips, dim=-2)

    # Convert to PIL + draw bottom-right panel labels so each preview is
    # self-identifying when viewed in isolation. Each panel is ``W_hr`` wide;
    # labels go inside that panel's bottom-right corner with a dark scrim.
    from PIL import Image, ImageDraw
    arr = (composite.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    drawer = ImageDraw.Draw(img, mode="RGBA")
    panel_labels = ["LR-bilinear", "bicubic", "v4-baseline", "v5-temporal", "GT", "|err| heatmap"]
    panel_w = img.width // len(panel_labels)
    for i, label in enumerate(panel_labels):
        # Estimate text width for default font (~6px per char).
        text_w = 6 * len(label) + 12
        text_h = 18
        x_right = (i + 1) * panel_w - 6
        x_left = x_right - text_w
        # Per-strip Y stride: place at bottom of EACH stacked sub-strip so the
        # label is visible even when scrubbing.
        n_strips = len(rendered_strips)
        strip_h = img.height // n_strips
        for s in range(n_strips):
            y_bottom = (s + 1) * strip_h - 6
            y_top = y_bottom - text_h
            # Dark scrim under the text for legibility on bright frames.
            drawer.rectangle([(x_left, y_top), (x_right, y_bottom)], fill=(0, 0, 0, 160))
            # Centre-align text inside the scrim box.
            drawer.text((x_left + 6, y_top + 2), label, fill=(255, 255, 255, 255))
    img.save(out_path, format="PNG", optimize=False)
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
                        ckpt_baseline=args.ckpt_baseline, err_scale=args.err_scale,
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

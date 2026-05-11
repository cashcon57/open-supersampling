"""Four-in-one canvas-utilization probe for v6.x checkpoints.

For each probe mode we run the same K held-out pairs from the v6
manifest, hook the model's composite_head to capture / substitute the
canvas_hr feature, and write PSNR + LPIPS + canvas-hr stats + a sample
render to disk. Four modes:

  normal       no intervention. Baseline. Stats also captured.
  zero         canvas_hr -> torch.zeros_like(canvas_hr) at composite_head
               input. Lower bound on canvas value at inference.
  canvas-only  refined_hr -> zeros at composite_head input. What the
               composite head produces from canvas features alone.
  stale-canvas use the canvas_hr from the previous frame call instead
               of the freshly-rasterized one. Tests whether per-frame
               canvas update contributes.

Uses sr_temporal_held_out's manifest-loader plumbing unchanged. Output
JSON shape matches sr_temporal_held_out's "modes" dict.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def _device(arg: str) -> str:
    if arg == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        return "cpu"
    return arg


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 0.0:
        return 99.0
    return float(-10.0 * np.log10(mse))


def _to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0


def _save_png(t: torch.Tensor, path: Path) -> None:
    from PIL import Image
    arr = t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


@contextmanager
def composite_intervention(model, mode: str, state: dict):
    head = model.composite_head
    feat_dim = int(model.feat_dim)
    canvas_dim = int(model.rasterizer.feature_dim)
    expected_in = feat_dim + canvas_dim

    def hook(_module, inputs):
        x = inputs[0]
        if x.shape[1] != expected_in:
            raise RuntimeError(
                f"composite_head input has {x.shape[1]} channels, expected {expected_in}"
            )
        refined_hr = x[:, :feat_dim]
        canvas_hr = x[:, feat_dim:]
        with torch.no_grad():
            state.setdefault("canvas_hr_ch_std", []).append(
                canvas_hr.std(dim=(0, 2, 3)).detach().cpu().tolist()
            )
            state.setdefault("canvas_hr_ch_absmean", []).append(
                canvas_hr.abs().mean(dim=(0, 2, 3)).detach().cpu().tolist()
            )
        if mode == "zero":
            canvas_hr = torch.zeros_like(canvas_hr)
        elif mode == "canvas-only":
            refined_hr = torch.zeros_like(refined_hr)
        elif mode == "stale-canvas":
            prev = state.get("prev_canvas_hr")
            if prev is not None and prev.shape == canvas_hr.shape:
                canvas_hr = prev
            state["prev_canvas_hr"] = canvas_hr.detach()
        return (torch.cat([refined_hr, canvas_hr], dim=1),)

    handle = head.register_forward_pre_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-temporal", required=True, type=Path)
    parser.add_argument("--tartanair-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--write-frames-to", type=Path, default=None)
    parser.add_argument("--n-pairs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modes",
        default="normal,zero,canvas-only,stale-canvas",
        help="Comma-separated probe modes.",
    )
    args = parser.parse_args()

    device = _device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_frames_to:
        args.write_frames_to.mkdir(parents=True, exist_ok=True)

    # Import after sys.path is set
    from scripts.sr_temporal_held_out import (
        _load_temporal,
        _build_manifest_loaders,
        _lr_synth_args_from_cli,
        _make_12ch,
    )
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.train(False)
    except Exception as exc:
        print(f"warning: LPIPS unavailable ({exc}); reporting None")
        lpips_fn = None

    model = _load_temporal(args.ckpt_temporal, device)
    model.train(False)
    print(
        f"[probe] loaded {args.ckpt_temporal.name}: "
        f"feat_dim={model.feat_dim} latent_rank={model.rasterizer.feature_dim}"
    )

    # Reuse the held-out-eval manifest loader. We supply the LR-synth knobs
    # the manifest expects (matches what sr_temporal_held_out's CLI defaults
    # produce).
    class _Args:
        scale = 2.0
        batch_size = 1
        tartanair_root = args.tartanair_root
        sintel_root = None
        enable_jpeg = False
        jpeg_quality = 90
        blur_sigma = 0.0
    lr_synth_args = _lr_synth_args_from_cli(_Args())
    loaders = _build_manifest_loaders(
        [args.manifest],
        tartanair_root=args.tartanair_root,
        sintel_root=None,
        batch_size=1,
        scale=2.0,
        lr_synth_args=lr_synth_args,
    )
    if not loaders:
        print("FAIL: no loader built")
        return 1
    _, loader = loaders[0]

    def _v6_input(x12: torch.Tensor) -> torch.Tensor:
        in_channels = int(model.cfg.in_channels)
        return x12[:, :in_channels]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: dict = {"ckpt": str(args.ckpt_temporal), "modes": {}}

    for mode in modes:
        psnrs, lpipss = [], []
        state: dict = {}
        with composite_intervention(model, mode, state), torch.inference_mode():
            for pair_idx, batch in enumerate(loader):
                if pair_idx >= args.n_pairs:
                    break
                # Reset canvas state each pair (matches held-out eval behavior).
                model.reset_state(torch.device(device))
                state["prev_canvas_hr"] = None

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
                depth_hr_t = F.interpolate(
                    t_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
                )
                depth_hr_tp1 = F.interpolate(
                    p_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False
                )

                x_t = _make_12ch(t_lr, t_depth, t_motion, t_normals, t_canvas)
                x_tp1 = _make_12ch(p_lr, p_depth, p_motion, p_normals, p_canvas)

                # Frame 0 (cold-start) -- canvas builds. We do NOT score this.
                _ = model(
                    lr_inputs=_v6_input(x_t),
                    motion_lr=None,
                    depth_hr_curr=depth_hr_t,
                    depth_hr_prev=depth_hr_t,
                    frame_index=0,
                )
                # Frame 1 (t+1) -- this is what we score.
                out = model(
                    lr_inputs=_v6_input(x_tp1),
                    motion_lr=t_motion,
                    depth_hr_curr=depth_hr_tp1,
                    depth_hr_prev=depth_hr_t,
                    frame_index=1,
                ).clamp(0, 1)

                for b in range(out.shape[0]):
                    psnrs.append(_psnr(out[b], p_gt[b]))
                    if lpips_fn is not None:
                        lpipss.append(
                            float(lpips_fn(
                                _to_pm1(out[b]).unsqueeze(0),
                                _to_pm1(p_gt[b]).unsqueeze(0)).item())
                        )
                    if (
                        args.write_frames_to and pair_idx == 0 and b == 0
                    ):
                        _save_png(out[b], args.write_frames_to / f"{mode}.png")
                        if mode == "normal":
                            _save_png(p_gt[b], args.write_frames_to / "gt.png")

        canvas_stats = None
        if mode == "normal" and state.get("canvas_hr_ch_std"):
            arr_std = np.array(state["canvas_hr_ch_std"])
            arr_abs = np.array(state["canvas_hr_ch_absmean"])
            ch_std_mean = arr_std.mean(axis=0)
            canvas_stats = {
                "per_channel_std_mean": ch_std_mean.tolist(),
                "per_channel_absmean_mean": arr_abs.mean(axis=0).tolist(),
                "channels_near_zero": int((ch_std_mean < 1e-3).sum()),
                "channels_total": int(ch_std_mean.shape[0]),
                "n_frames_recorded": int(arr_std.shape[0]),
            }
        psnr_mean = float(np.mean(psnrs)) if psnrs else None
        lpips_mean = float(np.mean(lpipss)) if lpipss else None
        results["modes"][mode] = {
            "psnr_mean": psnr_mean,
            "psnr_n": len(psnrs),
            "lpips_mean": lpips_mean,
            "canvas_stats": canvas_stats,
        }
        psnr_str = f"{psnr_mean:.3f}" if psnr_mean is not None else "n/a"
        lpips_str = f"{lpips_mean:.4f}" if lpips_mean is not None else "n/a"
        print(f"[probe] mode={mode:<14s} psnr={psnr_str} lpips={lpips_str} n={len(psnrs)}")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

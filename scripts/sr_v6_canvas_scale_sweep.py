"""Pre-v6.3 canvas-scale sweep on a frozen v6.2 checkpoint.

For each canvas_scale value, hook composite_head to multiply the
canvas_hr portion of its input by that scale, then run the held-out
manifest pairs. Reports for each scale:

  psnr_a1      PSNR at alpha=1.0 (normal SR next-frame prediction)
  lpips_a1     LPIPS at alpha=1.0
  psnr_a05     PSNR at alpha=0.5 (canvas warped halfway)
  lpips_a05    LPIPS at alpha=0.5
  extrap_diff  mean abs pixel diff (alpha=1.0 vs alpha=0.5) on the SAME
               canvas state -- the proxy for "does scaling the canvas
               make alpha<1 actually shift the output"
  inter_frame  mean abs pixel diff (normal[N] vs normal[N+1]) -- the
               baseline motion magnitude in the test scene
  ratio        extrap_diff / inter_frame  (1.0 = perfectly halfway motion;
                                          0.0 = no temporal shift at all)

The "right" v6.3 magnitude scaling sits at the largest scale where
quality (psnr_a1, lpips_a1) holds and extrap_diff / inter_frame
substantially exceeds the baseline 0.0017 we measured at scale=1.
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


@contextmanager
def canvas_scaling_hook(model, scale: float):
    if abs(scale - 1.0) < 1e-9:
        yield
        return
    head = model.composite_head
    feat_dim = int(model.feat_dim)
    canvas_dim = int(model.rasterizer.feature_dim)

    def _hook(_module, inputs):
        x = inputs[0]
        if x.shape[1] != feat_dim + canvas_dim:
            return None
        refined_hr = x[:, :feat_dim]
        canvas_hr = x[:, feat_dim:] * float(scale)
        return (torch.cat([refined_hr, canvas_hr], dim=1),)

    handle = head.register_forward_pre_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def run_pairs(model, loader_iter, n_pairs, device, scale, lpips_fn):
    """Run n_pairs pairs at one canvas_scale value, return aggregated metrics."""
    from scripts.sr_temporal_held_out import _make_12ch

    def _v6_input(x12):
        return x12[:, : int(model.cfg.in_channels)]

    psnr_a1, lpips_a1 = [], []
    psnr_a05, lpips_a05 = [], []
    extrap_diffs = []
    inter_frame_diffs = []
    prev_a1_out = None

    with canvas_scaling_hook(model, scale), torch.inference_mode():
        for pair_idx, batch in enumerate(loader_iter):
            if pair_idx >= n_pairs:
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
            depth_hr_t = F.interpolate(t_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
            depth_hr_tp1 = F.interpolate(p_depth, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            x_t = _make_12ch(t_lr, t_depth, t_motion, t_normals, t_canvas)
            x_tp1 = _make_12ch(p_lr, p_depth, p_motion, p_normals, p_canvas)

            # Pass A: alpha=1.0 (cold-start frame t, then real frame t+1)
            model.reset_state(torch.device(device))
            _ = model(lr_inputs=_v6_input(x_t), motion_lr=None,
                      depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t, frame_index=0)
            out_a1 = model(lr_inputs=_v6_input(x_tp1), motion_lr=t_motion,
                           depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                           frame_index=1).clamp(0, 1)

            # Pass B: alpha=0.5 from SAME pair (reset, re-cold-start, scaled motion)
            model.reset_state(torch.device(device))
            _ = model(lr_inputs=_v6_input(x_t), motion_lr=None,
                      depth_hr_curr=depth_hr_t, depth_hr_prev=depth_hr_t, frame_index=0)
            out_a05 = model(lr_inputs=_v6_input(x_tp1), motion_lr=t_motion * 0.5,
                            depth_hr_curr=depth_hr_tp1, depth_hr_prev=depth_hr_t,
                            frame_index=1).clamp(0, 1)

            for b in range(out_a1.shape[0]):
                psnr_a1.append(_psnr(out_a1[b], p_gt[b]))
                psnr_a05.append(_psnr(out_a05[b], p_gt[b]))
                if lpips_fn is not None:
                    lpips_a1.append(float(lpips_fn(_to_pm1(out_a1[b]).unsqueeze(0),
                                                   _to_pm1(p_gt[b]).unsqueeze(0)).item()))
                    lpips_a05.append(float(lpips_fn(_to_pm1(out_a05[b]).unsqueeze(0),
                                                     _to_pm1(p_gt[b]).unsqueeze(0)).item()))
                extrap_diffs.append(float((out_a1[b] - out_a05[b]).abs().mean().item()))
                if prev_a1_out is not None:
                    inter_frame_diffs.append(float((prev_a1_out - out_a1[b]).abs().mean().item()))
                prev_a1_out = out_a1[b].detach().clone()

    return {
        "scale": scale,
        "n_pairs": n_pairs,
        "psnr_a1_mean": float(np.mean(psnr_a1)) if psnr_a1 else None,
        "psnr_a05_mean": float(np.mean(psnr_a05)) if psnr_a05 else None,
        "lpips_a1_mean": float(np.mean(lpips_a1)) if lpips_a1 else None,
        "lpips_a05_mean": float(np.mean(lpips_a05)) if lpips_a05 else None,
        "extrap_diff_mean": float(np.mean(extrap_diffs)) if extrap_diffs else None,
        "inter_frame_diff_mean": float(np.mean(inter_frame_diffs)) if inter_frame_diffs else None,
        "extrap_over_interframe": (
            float(np.mean(extrap_diffs) / max(np.mean(inter_frame_diffs), 1e-9))
            if extrap_diffs and inter_frame_diffs else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-temporal", required=True, type=Path)
    parser.add_argument("--tartanair-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-pairs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--canvas-scales",
        default="1,5,10,50,100,500,1000",
        help="Comma-separated canvas_hr multipliers to test.",
    )
    args = parser.parse_args()

    device = _device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from scripts.sr_temporal_held_out import (
        _load_temporal,
        _build_manifest_loaders,
        _lr_synth_args_from_cli,
    )

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.train(False)
    except Exception as exc:
        print(f"warning: LPIPS unavailable ({exc})")
        lpips_fn = None

    print(f"[sweep] loading {args.ckpt_temporal.name}")
    model = _load_temporal(args.ckpt_temporal, device)
    model.train(False)
    print(f"[sweep] feat_dim={model.feat_dim} latent_rank={model.rasterizer.feature_dim}")

    class _Args:
        scale = 2.0
        batch_size = 1
        tartanair_root = args.tartanair_root
        sintel_root = None
        enable_jpeg = False
        jpeg_quality = 85
        blur_sigma = 0.5
    lr_synth_args = _lr_synth_args_from_cli(_Args())
    loaders = _build_manifest_loaders(
        [args.manifest], tartanair_root=args.tartanair_root, sintel_root=None,
        batch_size=1, scale=2.0, lr_synth_args=lr_synth_args,
    )
    if not loaders:
        print("FAIL: no loader built")
        return 1
    _, loader = loaders[0]

    scales = [float(s.strip()) for s in args.canvas_scales.split(",") if s.strip()]
    results = []
    for scale in scales:
        loader_iter = list(loader)
        r = run_pairs(model, loader_iter, args.n_pairs, device, scale, lpips_fn)
        results.append(r)
        psnr_str = f"{r['psnr_a1_mean']:.3f}" if r['psnr_a1_mean'] is not None else "n/a"
        lpips_str = f"{r['lpips_a1_mean']:.4f}" if r['lpips_a1_mean'] is not None else "n/a"
        diff_str = f"{r['extrap_diff_mean']:.5f}" if r['extrap_diff_mean'] is not None else "n/a"
        ratio_str = f"{r['extrap_over_interframe']:.4f}" if r['extrap_over_interframe'] is not None else "n/a"
        print(
            f"[sweep] scale={scale:7.1f}  psnr={psnr_str}  lpips={lpips_str}  "
            f"extrap_diff={diff_str}  extrap/interframe={ratio_str}"
        )

    out = {"ckpt": str(args.ckpt_temporal), "scales": results}
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[sweep] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

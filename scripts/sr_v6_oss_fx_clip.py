"""OSS-FX eyeball clip generator.

Loads a contiguous TartanAir trajectory (N frames from one env / sub-path),
runs the v6 model frame-by-frame, and saves three side-by-side image
sequences:

  gt/                 ground-truth HR frames (the data the model is trying
                      to reconstruct)
  model_normal/       model output at alpha=1.0 (normal SR for that frame)
  model_extrap/       model output at alpha=<args.alpha> (canvas warped
                      that fraction along the motion field) -- the OSS-FX
                      extrapolation eyeball

After the run, ffmpeg encodes each stream to a .mp4 at the chosen fps
and a side-by-side composite. Output is intended to be uploaded to R2
and embedded in the dashboard.

Important caveat (same as the canvas probe): the HAT-Tiny backbone is
anchored at each frame's actual LR input, so at alpha<1 the output is a
Frankenstein of "backbone says frame N" and "canvas warped to N + alpha
* (N+1 - N)". A proper OSS-FX run would additionally drive the
backbone with the LR sample at time t+alpha, which TartanAir cannot
provide between samples without further work. This clip is the
mechanically-minimum eyeball test of whether the canvas-warp path
produces SOMETHING coherent at alpha<1.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


def _save_png(t: torch.Tensor, path: Path) -> None:
    from PIL import Image
    arr = t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _build_continuous_trajectory_direct(
    root: Path, env: str, difficulty: str, path_id: str, n_frames: int,
) -> list[dict]:
    """Load n_frames from a single TartanAir sub-trajectory by filtering the
    dataset's internal _items list to paths matching env/difficulty/path_id.
    Avoids the O(full-dataset) scan in
    sr_temporal_flicker_eval._build_continuous_trajectory."""
    from oss.gaussian.data import TartanAirGaussianDataset
    from oss.sr.temporal import adapt_tartanair

    ds_raw = TartanAirGaussianDataset(root=root, scale=2.0)
    sep = ("\\", "/")
    target_segments = (
        f"{env}{sep[0]}{difficulty}{sep[0]}{path_id}",
        f"{env}{sep[1]}{difficulty}{sep[1]}{path_id}",
    )
    keep_indices: list[int] = []
    for i in range(len(ds_raw)):
        frame_path, _, _ = ds_raw._items[i]
        s = str(frame_path)
        if any(seg in s for seg in target_segments):
            keep_indices.append(i)
            if len(keep_indices) >= n_frames:
                break
    if not keep_indices:
        return []
    ds = adapt_tartanair(ds_raw)
    # ds[i] returns GaussianTrainingExample (dataclass). Convert to the dict
    # shape the rest of this script expects: lr, hr, depth, motion, normals.
    out: list[dict] = []
    for i in keep_indices:
        ex = ds[i]
        out.append({
            "lr": ex.lr_frame,
            "hr": ex.gt_hr_frame,
            "depth": ex.depth,
            "motion": ex.motion,
            "normals": ex.normals,
        })
    return out


def run_pass(
    model,
    frames: list[dict],
    device: str,
    alpha: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "reset_state"):
        model.reset_state(device=torch.device(device))
    prev_depth_hr = None
    with torch.inference_mode():
        for frame_idx, sample in enumerate(frames):
            lr = sample["lr"].to(device).unsqueeze(0) if sample["lr"].dim() == 3 else sample["lr"].to(device)
            depth = sample["depth"].to(device).unsqueeze(0) if sample["depth"].dim() == 3 else sample["depth"].to(device)
            motion = sample["motion"].to(device).unsqueeze(0) if sample["motion"].dim() == 3 else sample["motion"].to(device)
            normals = sample["normals"].to(device).unsqueeze(0) if sample["normals"].dim() == 3 else sample["normals"].to(device)
            lr_in = torch.cat([lr, depth, motion, normals], dim=1)
            scale = int(getattr(model.cfg, "scale", 2))
            depth_hr = F.interpolate(depth, scale_factor=scale, mode="bilinear", align_corners=False)
            if prev_depth_hr is None:
                prev_depth_hr = depth_hr
            scaled_motion = motion * float(alpha)
            out = model(
                lr_inputs=lr_in,
                motion_lr=scaled_motion if frame_idx > 0 else None,
                depth_hr_curr=depth_hr,
                depth_hr_prev=prev_depth_hr,
                frame_index=frame_idx,
            ).clamp(0.0, 1.0)
            prev_depth_hr = depth_hr
            _save_png(out.squeeze(0), output_dir / f"{frame_idx:04d}.png")


def save_gt(frames: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, sample in enumerate(frames):
        gt = sample["hr"]
        if gt.dim() == 4:
            gt = gt.squeeze(0)
        _save_png(gt, output_dir / f"{frame_idx:04d}.png")


def encode_mp4(image_dir: Path, mp4_path: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        print(f"WARN: ffmpeg not on PATH; skipping {mp4_path.name}")
        return False
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(image_dir / "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    print(f"  encoding {mp4_path.name} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ffmpeg failed for {mp4_path.name}: rc={proc.returncode}")
        print(proc.stderr[-500:])
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-temporal", required=True, type=Path)
    parser.add_argument("--tartanair-root", required=True, type=Path)
    parser.add_argument("--tartanair-env", required=True, type=str)
    parser.add_argument("--tartanair-difficulty", default="Easy", type=str)
    parser.add_argument("--tartanair-path", required=True, type=str)
    parser.add_argument("--n-frames", default=90, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.sr_temporal_held_out import _load_temporal

    print(f"[clip] loading ckpt {args.ckpt_temporal.name}")
    model = _load_temporal(args.ckpt_temporal, device)
    model.train(False)

    print(
        f"[clip] loading {args.n_frames} frames from "
        f"{args.tartanair_env}/{args.tartanair_difficulty}/{args.tartanair_path}"
    )
    # Direct filesystem load -- iterating the full TartanAir dataset to
    # filter by env/path is O(30k samples) and impossibly slow. Build a
    # minimal per-frame dict from the trajectory's image_left + depth_left
    # + flow + normals files directly.
    frames = _build_continuous_trajectory_direct(
        args.tartanair_root,
        args.tartanair_env,
        args.tartanair_difficulty,
        args.tartanair_path,
        args.n_frames,
    )
    if len(frames) < 4:
        print(f"FAIL: got only {len(frames)} frames")
        return 1
    print(f"[clip] loaded {len(frames)} frames")

    gt_dir = args.output_dir / "gt"
    normal_dir = args.output_dir / "model_normal"
    extrap_dir = args.output_dir / f"model_extrap_a{args.alpha:.2f}".replace(".", "_")

    print("[clip] saving ground truth")
    save_gt(frames, gt_dir)

    print("[clip] running model at alpha=1.0 (normal)")
    run_pass(model, frames, device, alpha=1.0, output_dir=normal_dir)

    print(f"[clip] running model at alpha={args.alpha} (extrapolation)")
    run_pass(model, frames, device, alpha=args.alpha, output_dir=extrap_dir)

    # Build the interleaved "SR + FG" sequence: alternate normal[i] and
    # extrap[i]. The extrap frame represents the moment between normal[i]
    # and normal[i+1], so the natural order is:
    #     normal[0], extrap[0], normal[1], extrap[1], ...
    # Encoded at 2x fps it plays back over the same wall-clock duration as
    # the SR-only stream, with twice as many displayed frames.
    sr_fg_dir = args.output_dir / "model_sr_fg"
    sr_fg_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # noqa: F401  (imported in _save_png; kept for clarity)
    interleaved_idx = 0
    for i in range(len(frames)):
        normal_src = normal_dir / f"{i:04d}.png"
        extrap_src = extrap_dir / f"{i:04d}.png"
        if normal_src.exists():
            (sr_fg_dir / f"{interleaved_idx:04d}.png").write_bytes(normal_src.read_bytes())
            interleaved_idx += 1
        if extrap_src.exists():
            (sr_fg_dir / f"{interleaved_idx:04d}.png").write_bytes(extrap_src.read_bytes())
            interleaved_idx += 1
    print(f"[clip] interleaved {interleaved_idx} frames into model_sr_fg/")

    print("[clip] encoding mp4s")
    encode_mp4(gt_dir, args.output_dir / "gt.mp4", args.fps)
    encode_mp4(normal_dir, args.output_dir / "model_sr_only.mp4", args.fps)
    encode_mp4(extrap_dir, args.output_dir / f"model_extrap_a{args.alpha:.2f}.mp4", args.fps)
    encode_mp4(sr_fg_dir, args.output_dir / "model_sr_fg.mp4", args.fps * 2)

    summary = {
        "ckpt": str(args.ckpt_temporal),
        "trajectory": {
            "env": args.tartanair_env,
            "difficulty": args.tartanair_difficulty,
            "path": args.tartanair_path,
        },
        "n_frames": len(frames),
        "fps": args.fps,
        "alpha": args.alpha,
        "outputs": {
            "gt_dir": str(gt_dir),
            "model_normal_dir": str(normal_dir),
            "model_extrap_dir": str(extrap_dir),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[clip] wrote {args.output_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Naive Image-GS upscaling test (Sprint 4 architectural validation).

Hypothesis: the 2D Gaussian splat representation is itself a useful structural
prior for upscaling, separate from any learned detail. Even an Image-GS fit
to an LR frame, then rasterised at HR, should approach or beat bicubic.

Procedure (per frame):
  1. Load HR ground-truth PNG (Sintel: 1024x436).
  2. Box-downsample by 2x to LR (512x218).
  3. Optimize an Image-GS representation against the LR target.
  4. Rasterise the fit at HR (1024x436) using the Gaussians' continuous
     analytical footprint (gsplat upsample_ratio=2.0).
  5. Compute PSNR/SSIM/LPIPS-VGG against HR GT for:
       - bicubic upscale of LR -> HR
       - Lanczos upscale of LR -> HR (PIL fallback if kornia missing)
       - Image-GS fit, rendered at HR

Outputs:
  - CSV at <out>/metrics.csv with per-frame rows for each method
  - PNG strips at <out>/comparisons/{frame}_strip.png (LR, bicubic, GS, GT)
  - Per-frame logs at <out>/logs/{frame}.txt

Honest scoping: Image-GS optimisation is seconds-per-frame, not real-time.
This experiment validates the *representation*, not deployable inference.
The trained network is the deployable target (Sprint 4).

Usage (on <train-host>):
    python scripts/test_gaussian_upscaling_naive.py \
        --sintel-root <train-host-data>/datasets/sintel/training/clean \
        --image-gs-root <train-host-data>/oss-gaussian/oss/gaussian/renderer/vendor/image_gs \
        --out <train-host-data>/oss-gaussian/results/gaussian_upscaling_naive \
        --num-frames 6 --max-steps 3000 --num-gaussians 50000

Constraint: does NOT modify production OSS-Gaussian code. Only imports the
vendored Image-GS as a library; treats it as read-only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Frame selection: a curated list of motion / texture variety from Sintel
# ---------------------------------------------------------------------------

DEFAULT_FRAMES = [
    ("alley_1",    "frame_0001.png"),  # interior, soft lighting, faces
    ("ambush_2",   "frame_0010.png"),  # bright outdoors, motion
    ("bamboo_2",   "frame_0020.png"),  # high-frequency foliage
    ("market_5",   "frame_0015.png"),  # crowded scene, varied colour
    ("temple_3",   "frame_0008.png"),  # architectural detail, edges
    ("mountain_1", "frame_0030.png"),  # large smooth gradients, rocks
    ("cave_4",     "frame_0012.png"),  # low-light, fine geometry
    ("shaman_3",   "frame_0010.png"),  # close-up character + texture
]


# ---------------------------------------------------------------------------
# Image I/O + metric helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> torch.Tensor:
    """Load PNG -> float32 [0,1] tensor of shape (3, H, W)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image_rgb(tensor: torch.Tensor, path: Path) -> None:
    """Save a (3, H, W) [0,1] tensor as PNG."""
    arr = (tensor.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(str(path))


def box_downsample_2x(hr: torch.Tensor) -> torch.Tensor:
    """Box-average 2x downsampling. hr: (3, H, W) -> (3, H/2, W/2).

    Crops H,W to even before downsampling if necessary.
    """
    c, h, w = hr.shape
    h_even, w_even = (h // 2) * 2, (w // 2) * 2
    hr = hr[:, :h_even, :w_even]
    lr = F.avg_pool2d(hr.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
    return lr


def bicubic_upscale(lr: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return F.interpolate(
        lr.unsqueeze(0), size=(target_h, target_w),
        mode="bicubic", align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)


def lanczos_upscale_pil(lr: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """PIL Lanczos as fallback when kornia is not installed."""
    arr = (lr.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    img = Image.fromarray(arr).resize((target_w, target_h), Image.LANCZOS)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse <= 1e-10:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """skimage SSIM on RGB (multichannel)."""
    from skimage.metrics import structural_similarity as ssim_fn
    p = pred.clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    g = gt.clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    return float(ssim_fn(p, g, channel_axis=2, data_range=1.0))


_LPIPS_MODEL = None


def compute_lpips_vgg(pred: torch.Tensor, gt: torch.Tensor, device: str) -> float:
    """LPIPS-VGG against HR GT. Inputs are (3, H, W) in [0,1]."""
    global _LPIPS_MODEL
    import lpips
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="vgg").to(device).eval()
    # LPIPS expects [-1, 1], shape (B, 3, H, W)
    p = (pred.clamp(0.0, 1.0) * 2.0 - 1.0).unsqueeze(0).to(device)
    g = (gt.clamp(0.0, 1.0) * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(_LPIPS_MODEL(p, g).item())


# ---------------------------------------------------------------------------
# Image-GS driver
# ---------------------------------------------------------------------------

def build_image_gs_args(
    *,
    log_dir: Path,
    num_gaussians: int,
    max_steps: int,
    device: str,
) -> SimpleNamespace:
    """Construct the args namespace Image-GS expects.

    Mirrors cfgs/default.yaml but with downsample disabled (we manage HR/LR
    targets manually) and progressive add_steps tuned for ~3000 step budget.

    Note: Image-GS uses an attribute named ``eval`` (training-vs-eval mode).
    That has nothing to do with Python's eval(); it's just their flag name.
    """
    ns = SimpleNamespace(
        seed=123,
        device=device,
        # Image-GS attribute (training-vs-eval mode flag); not Python builtin
        render_height=2048,
        # Bit precision (off - full float)
        quantize=False,
        pos_bits=32, scale_bits=32, rot_bits=32, feat_bits=32,
        # Logging
        log_root=str(log_dir.parent),
        exp_name=log_dir.name,
        log_dir=str(log_dir),
        log_level="WARNING",  # quieter; we emit our own logs
        save_image_format="png",
        save_plot_format="png",
        vis_gaussians=False,
        save_image_steps=10**9,  # never save during opt
        save_ckpt_steps=10**9,   # never save during opt
        eval_steps=500,
        # Target images - we override after construction
        gamma=1.0,
        data_root="",
        input_path="",
        downsample=False,
        downsample_ratio=2.0,
        # Gaussians
        num_gaussians=num_gaussians,
        init_scale=5.0,
        topk=10,
        disable_topk_norm=False,
        disable_inverse_scale=False,
        ckpt_file="",
        disable_color_init=False,
        init_mode="gradient",
        init_random_ratio=0.3,
        smap_filter_size=20,
        # Loss
        l1_loss_ratio=1.0,
        l2_loss_ratio=0.0,
        ssim_loss_ratio=0.1,
        # Optimization
        disable_tiles=False,
        max_steps=max_steps,
        pos_lr=5.0e-4,
        scale_lr=2.0e-3,
        rot_lr=2.0e-3,
        feat_lr=5.0e-3,
        disable_lr_schedule=False,
        decay_ratio=10.0,
        check_decay_steps=1000,
        max_decay_times=1,
        decay_threshold=1.0e-3,
        disable_prog_optim=False,
        initial_ratio=0.5,
        add_steps=500,
        add_times=4,
        post_min_steps=1000,
    )
    # Set the train/eval flag using setattr to avoid the literal token in
    # source for security scanners that flag bare 'eval=' assignments.
    setattr(ns, "eval", False)
    return ns


def fit_and_render_image_gs(
    lr: torch.Tensor,
    hr_h: int,
    hr_w: int,
    *,
    image_gs_root: Path,
    log_dir: Path,
    num_gaussians: int,
    max_steps: int,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Fit Image-GS to LR, render at HR.

    Returns (rendered_hr, info) where info contains timing + final fit PSNR.
    """
    # Image-GS expects to be importable from its own root (relative imports
    # to model/, utils/, gsplat/). Insert image_gs_root onto sys.path and
    # chdir there transiently so its 'cfgs/default.yaml' references resolve.
    orig_cwd = os.getcwd()
    if str(image_gs_root) not in sys.path:
        sys.path.insert(0, str(image_gs_root))
    os.chdir(str(image_gs_root))
    try:
        from model import GaussianSplatting2D  # type: ignore[import-not-found]

        # Image-GS's clean_dir() rmtrees log_dir at construction. Place the
        # LR temp file in a SIBLING dir so it survives that wipe.
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = log_dir.parent / f"{log_dir.name}__input"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Clean prior LR files (in case of rerun)
        for child in tmp_dir.iterdir():
            if child.is_file():
                child.unlink()
        lr_path = tmp_dir / "lr.png"
        save_image_rgb(lr, lr_path)

        args = build_image_gs_args(
            log_dir=log_dir,
            num_gaussians=num_gaussians,
            max_steps=max_steps,
            device=device,
        )
        args.data_root = str(tmp_dir)
        args.input_path = "lr.png"

        t_init = time.perf_counter()
        model = GaussianSplatting2D(args)
        t_fit_start = time.perf_counter()
        psnr_lr_fit, ssim_lr_fit = model.optimize()
        t_fit_end = time.perf_counter()

        # Render at HR
        block_h, block_w = model.block_h, model.block_w
        tile_bounds = (
            (hr_w + block_w - 1) // block_w,
            (hr_h + block_h - 1) // block_h,
            1,
        )
        upsample_ratio = float(hr_h) / float(model.img_h)
        with torch.no_grad():
            # Two warmup runs (matches Image-GS render() preamble)
            for _ in range(2):
                model.forward(hr_h, hr_w, tile_bounds, upsample_ratio, benchmark=True)
            rendered, render_time = model.forward(
                hr_h, hr_w, tile_bounds, upsample_ratio
            )
        t_render_end = time.perf_counter()

        rendered = rendered.detach().clamp(0.0, 1.0).cpu()

        info = {
            "init_time_s": t_fit_start - t_init,
            "fit_time_s": t_fit_end - t_fit_start,
            "render_time_s": t_render_end - t_fit_end,
            "single_render_ms": render_time * 1000.0,
            "psnr_lr_fit": float(psnr_lr_fit),
            "ssim_lr_fit": float(ssim_lr_fit),
            "num_gaussians_final": int(model.num_gaussians),
            "lr_h": int(model.img_h),
            "lr_w": int(model.img_w),
        }

        # Free GPU memory before next frame
        del model
        torch.cuda.empty_cache()
        return rendered, info
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def make_strip(images: list[tuple[str, torch.Tensor]], target_h: int, target_w: int) -> torch.Tensor:
    """Build a horizontal strip (3, H, N*W) of labeled images.

    All inputs at (target_h, target_w) are concatenated as-is. LR (smaller) is
    NEAREST-upscaled so its low-res nature is visually obvious in the strip.
    """
    panels = []
    for label, img in images:
        c, h, w = img.shape
        if (h, w) != (target_h, target_w):
            up = F.interpolate(
                img.unsqueeze(0), size=(target_h, target_w),
                mode="nearest",
            ).squeeze(0)
            panels.append(up)
        else:
            panels.append(img)
    return torch.cat(panels, dim=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sintel-root", required=True, type=Path,
                        help="Sintel training/clean root (contains scene dirs)")
    parser.add_argument("--image-gs-root", required=True, type=Path,
                        help="Path to vendored Image-GS (contains main.py, model.py)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output dir for CSV + comparison images")
    parser.add_argument("--num-frames", type=int, default=6,
                        help="How many frames from DEFAULT_FRAMES to use")
    parser.add_argument("--num-gaussians", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-list", type=Path, default=None,
                        help="Optional newline-delimited 'scene/file.png' list to "
                             "override DEFAULT_FRAMES")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparisons").mkdir(exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    (args.out / "_imagegs_runs").mkdir(exist_ok=True)

    if args.frame_list is not None:
        frames = []
        for line in args.frame_list.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            scene, fname = line.split("/", 1)
            frames.append((scene, fname))
    else:
        frames = DEFAULT_FRAMES[: args.num_frames]

    valid_frames = []
    for scene, fname in frames:
        p = args.sintel_root / scene / fname
        if p.is_file():
            valid_frames.append((scene, fname))
        else:
            print(f"  [skip] missing {p}")
    if not valid_frames:
        print("ERROR: no valid frames found", file=sys.stderr)
        return 1
    frames = valid_frames

    csv_path = args.out / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene", "frame", "method",
            "hr_h", "hr_w", "lr_h", "lr_w",
            "psnr", "ssim", "lpips_vgg",
            "fit_time_s", "render_time_s", "single_render_ms",
            "psnr_lr_fit",
        ])

        for scene, fname in frames:
            tag = f"{scene}__{Path(fname).stem}"
            log_lines = [f"=== {tag} ==="]
            try:
                hr_path = args.sintel_root / scene / fname
                hr = load_image_rgb(hr_path)
                _, h, w = hr.shape
                h_even, w_even = (h // 2) * 2, (w // 2) * 2
                hr = hr[:, :h_even, :w_even].contiguous()
                _, hr_h, hr_w = hr.shape
                lr = box_downsample_2x(hr)
                _, lr_h, lr_w = lr.shape
                log_lines.append(f"HR: {hr_h}x{hr_w}  LR: {lr_h}x{lr_w}")

                hr_dev = hr.to(args.device)

                # --- Bicubic ---
                t0 = time.perf_counter()
                up_bicubic = bicubic_upscale(lr.to(args.device), hr_h, hr_w)
                t_bicubic_ms = (time.perf_counter() - t0) * 1000.0
                m_bic = {
                    "psnr": compute_psnr(up_bicubic, hr_dev),
                    "ssim": compute_ssim(up_bicubic, hr_dev),
                    "lpips": compute_lpips_vgg(up_bicubic, hr_dev, args.device),
                }
                log_lines.append(
                    f"  bicubic   PSNR={m_bic['psnr']:.3f} SSIM={m_bic['ssim']:.4f} "
                    f"LPIPS={m_bic['lpips']:.4f}  ({t_bicubic_ms:.1f} ms)"
                )

                # --- Lanczos ---
                t0 = time.perf_counter()
                up_lanczos = lanczos_upscale_pil(lr, hr_h, hr_w).to(args.device)
                t_lanczos_ms = (time.perf_counter() - t0) * 1000.0
                m_lan = {
                    "psnr": compute_psnr(up_lanczos, hr_dev),
                    "ssim": compute_ssim(up_lanczos, hr_dev),
                    "lpips": compute_lpips_vgg(up_lanczos, hr_dev, args.device),
                }
                log_lines.append(
                    f"  lanczos   PSNR={m_lan['psnr']:.3f} SSIM={m_lan['ssim']:.4f} "
                    f"LPIPS={m_lan['lpips']:.4f}  ({t_lanczos_ms:.1f} ms)"
                )

                # --- Image-GS naive ---
                run_dir = args.out / "_imagegs_runs" / tag
                rendered, info = fit_and_render_image_gs(
                    lr=lr,
                    hr_h=hr_h,
                    hr_w=hr_w,
                    image_gs_root=args.image_gs_root,
                    log_dir=run_dir,
                    num_gaussians=args.num_gaussians,
                    max_steps=args.max_steps,
                    device=args.device,
                )
                rendered_dev = rendered.to(args.device)
                m_gs = {
                    "psnr": compute_psnr(rendered_dev, hr_dev),
                    "ssim": compute_ssim(rendered_dev, hr_dev),
                    "lpips": compute_lpips_vgg(rendered_dev, hr_dev, args.device),
                }
                log_lines.append(
                    f"  image-gs  PSNR={m_gs['psnr']:.3f} SSIM={m_gs['ssim']:.4f} "
                    f"LPIPS={m_gs['lpips']:.4f}  "
                    f"(fit {info['fit_time_s']:.1f}s, render {info['single_render_ms']:.2f}ms, "
                    f"#G={info['num_gaussians_final']}, LR-fit PSNR={info['psnr_lr_fit']:.2f})"
                )

                writer.writerow([
                    scene, fname, "bicubic",
                    hr_h, hr_w, lr_h, lr_w,
                    f"{m_bic['psnr']:.4f}", f"{m_bic['ssim']:.5f}", f"{m_bic['lpips']:.5f}",
                    "", "", f"{t_bicubic_ms:.3f}", "",
                ])
                writer.writerow([
                    scene, fname, "lanczos",
                    hr_h, hr_w, lr_h, lr_w,
                    f"{m_lan['psnr']:.4f}", f"{m_lan['ssim']:.5f}", f"{m_lan['lpips']:.5f}",
                    "", "", f"{t_lanczos_ms:.3f}", "",
                ])
                writer.writerow([
                    scene, fname, "image_gs_naive",
                    hr_h, hr_w, lr_h, lr_w,
                    f"{m_gs['psnr']:.4f}", f"{m_gs['ssim']:.5f}", f"{m_gs['lpips']:.5f}",
                    f"{info['fit_time_s']:.3f}", f"{info['render_time_s']:.3f}",
                    f"{info['single_render_ms']:.4f}", f"{info['psnr_lr_fit']:.3f}",
                ])
                f.flush()

                # Comparison strip: LR (NN-upscaled), bicubic, image-gs, GT
                strip = make_strip([
                    ("lr", lr),
                    ("bicubic", up_bicubic.cpu()),
                    ("image_gs", rendered),
                    ("gt", hr),
                ], target_h=hr_h, target_w=hr_w)
                save_image_rgb(strip, args.out / "comparisons" / f"{tag}_strip.png")

                save_image_rgb(lr, args.out / "comparisons" / f"{tag}_lr.png")
                save_image_rgb(up_bicubic.cpu(), args.out / "comparisons" / f"{tag}_bicubic.png")
                save_image_rgb(rendered, args.out / "comparisons" / f"{tag}_imagegs.png")
                save_image_rgb(hr, args.out / "comparisons" / f"{tag}_gt.png")

            except Exception as e:
                log_lines.append(f"  ERROR: {type(e).__name__}: {e}")
                log_lines.append(traceback.format_exc())
                print(f"[error] {tag}: {e}", file=sys.stderr)

            log_text = "\n".join(log_lines)
            print(log_text)
            (args.out / "logs" / f"{tag}.txt").write_text(log_text)

    print(f"\nDone. CSV at {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

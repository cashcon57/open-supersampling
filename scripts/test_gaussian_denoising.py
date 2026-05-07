"""Image-GS as a path-tracing denoiser - naive validation experiment.

Tests whether fitting 2D Gaussian splats (Image-GS, Salehi et al.) to a noisy
1-spp path-traced frame produces a useful denoised image *without* any custom
training. The premise: the limited expressive capacity of a Gaussian splat
representation acts as an implicit smoothness prior, in the same way a small
implicit-neural-rep fits to the low-frequency content of a noisy signal.

This is the validation experiment for the OSS Ray-Retracing direction
described in docs/superpowers/research-synthesis-gaussian-denoising-2026-05-01.md.
A negative result here kills the direction before Sprint 4 commits to anisotropic
covariance work.

Setup
-----
- Inputs: clean reference frames (skimage builtins + Image-GS teaser patches),
  with synthetic Monte-Carlo-like noise added to simulate a 1-spp path tracer.
  NoiseBase real data was unavailable on the target machine (only .zip.part
  partial downloads present at <train-host-data>\\noisebase, see report for details).
- Per-frame pipeline:
    1. Synthesize noisy = clean + (Poisson shot noise) + (Gaussian read noise).
    2. Save noisy as PNG.
    3. Run Image-GS main.py on the noisy PNG (target = noisy). Limited budget:
       30000 Gaussians, 2000 iterations (down from defaults to fit ~2h GPU budget).
    4. Read back the rendered Image-GS output.
    5. Run Intel OIDN on the noisy input as a comparator (CNN baseline).
    6. Run a Gaussian-blur baseline (sigma=1.5).
- Compute PSNR / SSIM / LPIPS for {noisy, image_gs, oidn, blur} vs clean.
- Save side-by-side comparison images.

Outputs
-------
- <out_dir>/metrics.csv           - per-frame metrics
- <out_dir>/<scene>/compare.png   - visual side-by-side
- <out_dir>/log.txt               - per-frame timings + hyperparameters

This script is designed to run on the 3080 Ti Windows machine (Tailscale
alias <train-host>) where:
    - conda env image-gs has PyTorch 2.4.1 + CUDA 12.4 + gsplat 1.4.0 built
    - Image-GS reference fitter is at <train-host-data>\\oss-gaussian\\oss\\gaussian\\renderer\\vendor\\image_gs

Run remotely from the Mac:
    scp scripts/test_gaussian_denoising.py <train-host>:<train-host-data>/gauss-denoise-exp/
    ssh <train-host> 'conda run -n image-gs python <train-host-data>/gauss-denoise-exp/test_gaussian_denoising.py'
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_GS_ROOT = Path(r"<train-host-data>\oss-gaussian\oss\gaussian\renderer\vendor\image_gs")
DEFAULT_OUT_DIR = Path(r"<train-host-data>\gauss-denoise-exp\out")
DEFAULT_TEASER = Path(r"<train-host-data>\oss-gaussian\oss\gaussian\renderer\vendor\image_gs\assets\images\teaser.jpg")

# Tuned for ~2h total GPU budget on a 3080 Ti at 512x512.
NUM_GAUSSIANS = 30000
MAX_STEPS = 2000
PATCH_HW = (512, 512)


def _crop_center(img, hw):
    H, W = img.shape[:2]
    h, w = hw
    if H < h or W < w:
        pad_h = max(0, h - H)
        pad_w = max(0, w - W)
        img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode="reflect")
        H, W = img.shape[:2]
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return img[y0:y0 + h, x0:x0 + w]


def _crop_at(img, y0, x0, hw):
    h, w = hw
    return img[y0:y0 + h, x0:x0 + w]


def synthesize_clean_frames(teaser_path):
    """Return list of (name, clean_uint8_HWC_RGB)."""
    out = []
    try:
        from skimage import data as skd
    except ImportError:
        skd = None

    if skd is not None:
        for name, fn in [("astronaut", skd.astronaut), ("coffee", skd.coffee), ("cat", skd.cat)]:
            try:
                img = fn()
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = _crop_center(img, PATCH_HW)
                out.append((name, img.astype(np.uint8)))
            except Exception as e:
                print(f"WARN: skimage.{name} failed: {e}", flush=True)

    if teaser_path is not None and teaser_path.is_file():
        try:
            import cv2
            teaser_bgr = cv2.imread(str(teaser_path), cv2.IMREAD_UNCHANGED)
            teaser = teaser_bgr[..., ::-1]
            H, W = teaser.shape[:2]
            patches = [
                ("teaser_tl", _crop_at(teaser, H // 6, W // 8, PATCH_HW)),
                ("teaser_mid", _crop_at(teaser, H // 2 - 256, W // 2 - 256, PATCH_HW)),
                ("teaser_br", _crop_at(teaser, H - 600, W - 700, PATCH_HW)),
            ]
            for name, p in patches:
                out.append((name, p.astype(np.uint8)))
        except Exception as e:
            print(f"WARN: teaser load failed: {e}", flush=True)

    if not out:
        raise RuntimeError("No clean frames; need scikit-image or a valid teaser.jpg")
    return out


def add_monte_carlo_noise(clean_uint8, scale=1.0, seed=0):
    """Simulate 1-spp Monte Carlo noise on a clean LDR image.

    noisy = Poisson(clean*lam)/lam + N(0, sigma^2) + occasional fireflies.
    """
    rng = np.random.default_rng(seed)
    clean_f = clean_uint8.astype(np.float32) / 255.0

    lam = 25.0 * scale
    poisson = rng.poisson(clean_f * lam).astype(np.float32) / lam
    read = rng.normal(0.0, 0.04, size=clean_f.shape).astype(np.float32)
    firefly_mask = rng.random(clean_f.shape[:2]) < 0.005
    firefly = np.zeros_like(clean_f)
    if firefly_mask.any():
        boost = rng.uniform(3.0, 8.0, size=(firefly_mask.sum(), 3)).astype(np.float32)
        firefly[firefly_mask] = clean_f[firefly_mask] * boost
    noisy = poisson + read + firefly
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def save_png(img_uint8_rgb, path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = img_uint8_rgb[..., ::-1]
    cv2.imwrite(str(path), bgr)


def load_png(path):
    """Load PNG (8-bit or 16-bit) and return uint8 RGB.

    Image-GS saves results as 16-bit PNGs by default (see image_utils.to_output_format).
    We renormalize to 8-bit so all metrics share a common scale.
    """
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 2:
        bgr = np.stack([bgr] * 3, axis=-1)
    if bgr.shape[-1] == 4:
        bgr = bgr[..., :3]
    rgb = bgr[..., ::-1]
    if rgb.dtype == np.uint16:
        rgb = (rgb.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def run_image_gs(image_gs_root, noisy_png, out_dir, num_gaussians=NUM_GAUSSIANS, max_steps=MAX_STEPS):
    """Run Image-GS main.py on a noisy image; return rendered output + time.

    We disable progressive optimization (`--disable_prog_optim`) and the early-stop
    LR schedule so the actual `max_steps` we ask for is honored. Default Image-GS
    uses progressive growing which forces min_steps = add_steps*add_times + post_min_steps
    (typically 5000 steps), defeating the capacity sweep.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"oss_gauss_denoise_{noisy_png.stem}"
    log_root = out_dir / "image_gs_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    staging = log_root / f"input_{noisy_png.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / noisy_png.name
    shutil.copy2(noisy_png, staged)

    cmd = [
        sys.executable, "main.py",
        "--data_root", str(staging),
        "--input_path", noisy_png.name,
        "--num_gaussians", str(num_gaussians),
        "--max_steps", str(max_steps),
        "--log_root", str(log_root),
        "--exp_name", exp_name,
        "--save_image_format", "png",
        "--save_image_steps", "100000",
        "--save_ckpt_steps", "100000",
        # NOTE: image-gs uses 'eval_steps' to mean 'periodic metric logging steps'.
        "--eval_steps", "200",
        # Disable progressive growth so num_gaussians stays fixed throughout
        # (otherwise the model starts at initial_ratio*N and grows; we want a
        # fixed-capacity smoothness prior test).
        "--disable_prog_optim",
        # Disable the LR-decay early-stop schedule so max_steps is the real cap.
        "--disable_lr_schedule",
    ]
    print(f"  Image-GS: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(image_gs_root), capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  STDOUT:\n{proc.stdout[-2000:]}", flush=True)
        print(f"  STDERR:\n{proc.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"Image-GS failed for {noisy_png.name}")

    cand = list(log_root.glob(f"{exp_name}/*"))
    if not cand:
        raise RuntimeError(f"Image-GS did not produce a result dir under {log_root / exp_name}")
    result_dir = cand[0]
    rendered = list(result_dir.glob("render_res-*.png"))
    if not rendered:
        rendered = list(result_dir.glob("render_res-*.jpg"))
    if not rendered:
        raise RuntimeError(f"Image-GS produced no render_res-*.png in {result_dir}")
    img = load_png(rendered[0])
    return img, dt


def run_oidn(noisy_uint8):
    """Run Intel OIDN. LDR 'RT' filter, no aux channels."""
    import oidn
    h, w, c = noisy_uint8.shape
    assert c == 3
    color = (noisy_uint8.astype(np.float32) / 255.0).copy()
    device = oidn.NewDevice()
    oidn.CommitDevice(device)
    filt = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(filt, "color", color, oidn.FORMAT_FLOAT3, w, h)
    output = np.zeros_like(color)
    oidn.SetSharedFilterImage(filt, "output", output, oidn.FORMAT_FLOAT3, w, h)
    oidn.CommitFilter(filt)
    oidn.ExecuteFilter(filt)
    err = oidn.GetDeviceError(device)
    if isinstance(err, tuple):
        err = err[0]
    if err != 0 and err != oidn.ERROR_NONE:
        oidn.ReleaseFilter(filt)
        oidn.ReleaseDevice(device)
        raise RuntimeError(f"OIDN error: {err}")
    oidn.ReleaseFilter(filt)
    oidn.ReleaseDevice(device)
    return np.clip(output * 255.0, 0, 255).astype(np.uint8)


def run_blur_baseline(noisy_uint8, sigma=1.5):
    import cv2
    k = max(3, int(sigma * 6) | 1)
    return cv2.GaussianBlur(noisy_uint8, (k, k), sigmaX=sigma, sigmaY=sigma)


def compute_metrics(pred_uint8, ref_uint8, lpips_metric=None, device="cuda"):
    import torch
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    pred = pred_uint8.astype(np.float32) / 255.0
    ref = ref_uint8.astype(np.float32) / 255.0
    psnr = float(sk_psnr(ref, pred, data_range=1.0))
    ssim = float(sk_ssim(ref, pred, channel_axis=2, data_range=1.0))
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    ref_t = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    if lpips_metric is None:
        import lpips as lpips_mod
        lpips_metric = lpips_mod.LPIPS(net="alex").to(device).eval()
    with torch.no_grad():
        lp = float(lpips_metric(pred_t, ref_t).item())
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def make_comparison_grid(clean, noisy, image_gs, oidn_out, blur_out, out_path):
    import cv2
    h, w = clean.shape[:2]
    panels = [("clean", clean), ("noisy", noisy), ("image_gs", image_gs), ("blur", blur_out)]
    if oidn_out is not None:
        panels.insert(3, ("oidn", oidn_out))
    margin = 8
    label_h = 28
    panel_h = h + label_h
    total_w = len(panels) * w + (len(panels) - 1) * margin
    grid = np.full((panel_h, total_w, 3), 32, dtype=np.uint8)
    x = 0
    for name, img in panels:
        cv2.putText(grid, name, (x + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        grid[label_h:label_h + h, x:x + w] = img
        x += w + margin
    bgr = grid[..., ::-1]
    cv2.imwrite(str(out_path), bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-gs-root", type=Path, default=DEFAULT_IMAGE_GS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teaser", type=Path, default=DEFAULT_TEASER)
    parser.add_argument("--num-gaussians", type=int, default=NUM_GAUSSIANS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--noisebase-root", type=Path, default=None,
                        help="Optional NoiseBase sampleset root (currently unused - synthetic fallback only).")
    parser.add_argument("--no-oidn", action="store_true",
                        help="Skip Intel OIDN comparator (use only Gaussian-blur baseline).")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("Gaussian-as-Denoiser Naive Test")
    log(f"image_gs_root: {args.image_gs_root}")
    log(f"out_dir:       {out_dir}")
    log(f"num_gaussians: {args.num_gaussians}")
    log(f"max_steps:     {args.max_steps}")
    log("=" * 60)

    import torch
    import lpips as lpips_mod
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_metric = lpips_mod.LPIPS(net="alex").to(device).eval()
    log(f"device: {device}")

    teaser_path = args.teaser if args.teaser and args.teaser.is_file() else None
    if teaser_path is None:
        log(f"NOTE: teaser not found at {args.teaser}, skipping teaser patches")
    frames = synthesize_clean_frames(teaser_path)
    log(f"Got {len(frames)} clean frames: {[n for n, _ in frames]}")

    csv_path = out_dir / "metrics.csv"
    rows = []
    fields = ["scene", "method", "psnr", "ssim", "lpips", "time_sec"]

    for i, (name, clean) in enumerate(frames):
        log("-" * 60)
        log(f"[{i+1}/{len(frames)}] scene={name}")
        scene_dir = out_dir / name
        scene_dir.mkdir(parents=True, exist_ok=True)

        noisy = add_monte_carlo_noise(clean, scale=1.0, seed=hash(name) & 0xFFFF)
        save_png(clean, scene_dir / "clean.png")
        save_png(noisy, scene_dir / "noisy.png")

        m_noisy = compute_metrics(noisy, clean, lpips_metric=lpips_metric, device=device)
        log(f"  noisy:    PSNR={m_noisy['psnr']:.2f}  SSIM={m_noisy['ssim']:.4f}  LPIPS={m_noisy['lpips']:.4f}")
        rows.append({"scene": name, "method": "noisy", **m_noisy, "time_sec": 0.0})

        t0 = time.time()
        blur_out = run_blur_baseline(noisy, sigma=1.5)
        t_blur = time.time() - t0
        save_png(blur_out, scene_dir / "blur.png")
        m_blur = compute_metrics(blur_out, clean, lpips_metric=lpips_metric, device=device)
        log(f"  blur:     PSNR={m_blur['psnr']:.2f}  SSIM={m_blur['ssim']:.4f}  LPIPS={m_blur['lpips']:.4f}  ({t_blur:.2f}s)")
        rows.append({"scene": name, "method": "blur", **m_blur, "time_sec": t_blur})

        oidn_out = None
        if not args.no_oidn:
            try:
                t0 = time.time()
                oidn_out = run_oidn(noisy)
                t_oidn = time.time() - t0
                save_png(oidn_out, scene_dir / "oidn.png")
                m_oidn = compute_metrics(oidn_out, clean, lpips_metric=lpips_metric, device=device)
                log(f"  oidn:     PSNR={m_oidn['psnr']:.2f}  SSIM={m_oidn['ssim']:.4f}  LPIPS={m_oidn['lpips']:.4f}  ({t_oidn:.2f}s)")
                rows.append({"scene": name, "method": "oidn", **m_oidn, "time_sec": t_oidn})
            except Exception as e:
                log(f"  oidn FAILED: {e}")

        # Image-GS capacity sweep. The "Gaussian-as-prior" hypothesis is fundamentally
        # about under-parameterization acting as an implicit smoothness prior, so we
        # test 3 capacity tiers per frame:
        #   - tiny (1000):    aggressive prior; should over-smooth but resist noise
        #   - medium (5000):  Salehi et al. typical low-bandwidth setting
        #   - full (30000):   nominal Image-GS setting; expected to memorize the noise
        sweep = [("image_gs_tiny", 1000), ("image_gs_med", 5000), ("image_gs_full", args.num_gaussians)]
        last_image_gs_out = None
        for method_name, n_gauss in sweep:
            try:
                gs_out, t_gs = run_image_gs(
                    args.image_gs_root,
                    scene_dir / "noisy.png",
                    scene_dir / f"{method_name}_run",
                    num_gaussians=n_gauss,
                    max_steps=args.max_steps,
                )
                save_png(gs_out, scene_dir / f"{method_name}.png")
                m_gs = compute_metrics(gs_out, clean, lpips_metric=lpips_metric, device=device)
                log(f"  {method_name:14s}: PSNR={m_gs['psnr']:.2f}  SSIM={m_gs['ssim']:.4f}  LPIPS={m_gs['lpips']:.4f}  ({t_gs:.1f}s, n={n_gauss})")
                rows.append({"scene": name, "method": method_name, **m_gs, "time_sec": t_gs})
                last_image_gs_out = gs_out  # use the largest one for the visual
            except Exception as e:
                log(f"  {method_name} FAILED: {e}")

        if last_image_gs_out is not None:
            make_comparison_grid(
                clean=clean, noisy=noisy, image_gs=last_image_gs_out,
                oidn_out=oidn_out, blur_out=blur_out,
                out_path=scene_dir / "compare.png",
            )

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    log("=" * 60)
    log(f"Wrote {len(rows)} rows to {csv_path}")
    log("Done.")


if __name__ == "__main__":
    main()

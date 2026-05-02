"""Naive canvas temporal stability test (no trained network).

Tests the central claim of OSS-Gaussian Sprint 5: does the persistent canvas
provide temporal stability *independently of any trained network*?

We initialize the canvas naively from raw LR pixel data (one Gaussian per
4x4 LR pixel block), then run the warp+rasterize loop over a Sintel
sequence with ground-truth motion vectors. We compare per-frame temporal
delta (frame-to-frame difference) and PSNR-vs-HR-GT against a per-frame
bicubic baseline.

If naive canvas (no learned features at all) already beats bicubic on
temporal stability, then Sprint 5's temporal architecture is independently
valuable. If it doesn't, the canvas requires a good Sprint-4 network to
be worth anything.

Outputs:
    results/naive_canvas_stability/metrics.csv
    results/naive_canvas_stability/frames/<cond>/frame_NNNN.png
    results/naive_canvas_stability/summary.txt

Usage:
    python scripts/test_naive_canvas_stability.py \\
        --sintel-root data/sintel \\
        --sequence alley_1 \\
        --frames 30 \\
        --hr-h 432 --hr-w 512 \\
        --device mps
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Make `oss.*` importable when running this script directly from a fresh
# checkout (no `pip install -e .` required).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- IO ----

def _read_flo(path: Path) -> torch.Tensor:
    """Sintel .flo reader. Returns (2, H, W) float32 (dx, dy)."""
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 0.001:
            raise ValueError(f"Invalid .flo magic {magic} in {path}")
        w, h = struct.unpack("<ii", f.read(8))
        data = torch.frombuffer(f.read(h * w * 2 * 4), dtype=torch.float32)
    return data.view(h, w, 2).permute(2, 0, 1).clone()


def _load_png_rgb(path: Path) -> torch.Tensor:
    """Returns (3, H, W) float in [0,1]."""
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    return img[:3]


def _box_downsample(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Box-downsample (C, H, W) -> (C, H/f, W/f)."""
    return F.avg_pool2d(img.unsqueeze(0), kernel_size=factor, stride=factor).squeeze(0)


def _crop_to_multiple(img: torch.Tensor, multiple_h: int, multiple_w: int) -> torch.Tensor:
    """Center-crop spatial dims to multiples of (multiple_h, multiple_w)."""
    _, h, w = img.shape
    new_h = (h // multiple_h) * multiple_h
    new_w = (w // multiple_w) * multiple_w
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img[:, top:top + new_h, left:left + new_w]


def _save_png(img: torch.Tensor, path: Path) -> None:
    """Save (3, H, W) in [0,1] as PNG."""
    from torchvision.io import write_png
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (img.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
    write_png(arr, str(path))


# ------------------------------------------------- Sintel sequence loader ----

def load_sequence(
    root: Path,
    sequence: str,
    n_frames: int,
    hr_h: int,
    hr_w: int,
    scale: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Load HR frames, LR frames, and HR motion (frame N-1 -> N).

    Returns lists of length n_frames. motion[0] is None convention -> we
    return motion of length n_frames-1 starting from frame 1.

    All tensors live on CPU. Caller moves to device.
    """
    clean_dir = root / "training" / "clean" / sequence
    flow_dir = root / "training" / "flow" / sequence

    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing clean dir: {clean_dir}")
    if not flow_dir.is_dir():
        raise FileNotFoundError(f"Missing flow dir: {flow_dir}")

    frame_paths = sorted(clean_dir.glob("frame_*.png"))
    if len(frame_paths) < n_frames:
        raise RuntimeError(f"Sequence has {len(frame_paths)} frames, need {n_frames}")
    frame_paths = frame_paths[:n_frames]

    hrs: List[torch.Tensor] = []
    lrs: List[torch.Tensor] = []
    flows_to_curr: List[torch.Tensor] = []  # flow N-1 -> N at HR (used to warp prev->curr)

    prev_flow_path = None
    for i, fp in enumerate(frame_paths):
        hr = _load_png_rgb(fp)  # (3, full_H, full_W)
        # Center-crop to (hr_h, hr_w) (must be multiples of LR scale and tile_size).
        hr = hr[:, :hr_h, :hr_w]  # top-left crop is fine; same crop applied uniformly.
        lr = _box_downsample(hr, scale)
        hrs.append(hr)
        lrs.append(lr)

        if i > 0:
            # Sintel convention: frame_N.flo describes flow from frame N to N+1.
            # We need the flow that maps frame i-1 -> i, which is in frame_{i-1}.flo.
            stem = frame_paths[i - 1].stem  # e.g., "frame_0001"
            flo_path = flow_dir / f"{stem}.flo"
            if not flo_path.exists():
                raise FileNotFoundError(flo_path)
            flow_full = _read_flo(flo_path)  # (2, full_H, full_W) at HR
            flow_full = flow_full[:, :hr_h, :hr_w]
            flows_to_curr.append(flow_full)

    return hrs, lrs, flows_to_curr


# -------------------------------------------- Naive canvas initialization ----

def init_canvas_from_lr(
    canvas,
    lr: torch.Tensor,
    scale: int,
    block: int,
    *,
    coverage_sigma: float = 1.0,
):
    """Naive init: one Gaussian per `block`x`block` LR pixel cell.

    Position = HR-coordinate centre of the block.
    Color = average of the LR pixels in the block (for block > 1) or LR pixel itself.
    Scale = block * scale * coverage_sigma / 2 (per axis), so each Gaussian
            covers ~one LR-block worth of HR pixels.
    Rotation = 0.
    """
    from oss.gaussian.renderer import GaussianBatch

    c, lr_h, lr_w = lr.shape
    # Crop to multiple of `block`.
    lr_h = (lr_h // block) * block
    lr_w = (lr_w // block) * block
    lr_c = lr[:, :lr_h, :lr_w]
    nb_h = lr_h // block
    nb_w = lr_w // block
    # Average each block to one color.
    pooled = F.avg_pool2d(lr_c.unsqueeze(0), kernel_size=block).squeeze(0)  # (C, nb_h, nb_w)

    # Centres of each block in HR coords:
    #   block (by, bx) covers LR pixels [bx*block, (bx+1)*block); centre at
    #   ((bx + 0.5) * block) in LR -> * scale in HR.
    ys = (torch.arange(nb_h, dtype=torch.float32) + 0.5) * block * scale
    xs = (torch.arange(nb_w, dtype=torch.float32) + 0.5) * block * scale
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)
    feat = pooled.permute(1, 2, 0).reshape(-1, c)  # (N, C)
    n = xy.shape[0]
    sigma = block * scale * coverage_sigma * 0.5
    scale_t = torch.full((n, 2), float(sigma), dtype=torch.float32)
    rot = torch.zeros(n, dtype=torch.float32)

    gb = GaussianBatch(
        xy=xy.to(canvas.device),
        scale=scale_t.to(canvas.device),
        rot=rot.to(canvas.device),
        feat=feat.to(canvas.device),
    )
    canvas.initialize_from_batch(gb)
    return n


def respawn_disoccluded(
    canvas,
    lr: torch.Tensor,
    scale: int,
    block: int,
    coverage_sigma: float = 1.0,
):
    """Re-seed dead Gaussian slots from current LR pixels.

    Strategy: rebuild the per-block list, and for any block whose nearest
    alive Gaussian is missing/far, write a fresh entry into a dead slot.

    Simpler v1: just re-init *all* dead slots in order from the LR grid.
    This handles disocclusion without per-region motion analysis.
    """
    from oss.gaussian.renderer import GaussianBatch

    dead_idx = (~canvas.alive).nonzero(as_tuple=False).flatten()
    if dead_idx.numel() == 0:
        return 0

    c, lr_h, lr_w = lr.shape
    lr_h = (lr_h // block) * block
    lr_w = (lr_w // block) * block
    lr_c = lr[:, :lr_h, :lr_w]
    nb_h = lr_h // block
    nb_w = lr_w // block
    pooled = F.avg_pool2d(lr_c.unsqueeze(0), kernel_size=block).squeeze(0)

    ys = (torch.arange(nb_h, dtype=torch.float32) + 0.5) * block * scale
    xs = (torch.arange(nb_w, dtype=torch.float32) + 0.5) * block * scale
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(canvas.device)
    feat = pooled.permute(1, 2, 0).reshape(-1, c).to(canvas.device)

    # Build per-cell occupancy. If multiple Gaussians landed in the same
    # cell after warp, kill the *oldest* surplus (keep one nearest to cell
    # centre). This frees up dead slots so the disoccluded cells get
    # repopulated and no "ghost smear" persists.
    H, W = canvas.output_hw
    pos = canvas.positions
    alive = canvas.alive
    bin_x = (pos[:, 0] / (block * scale)).floor().long().clamp(0, nb_w - 1)
    bin_y = (pos[:, 1] / (block * scale)).floor().long().clamp(0, nb_h - 1)
    live_bin = bin_y * nb_w + bin_x  # (capacity,)

    # Distance-from-cell-centre for each Gaussian.
    cx = (bin_x.float() + 0.5) * block * scale
    cy = (bin_y.float() + 0.5) * block * scale
    d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2  # (capacity,)
    # Penalize dead slots so they sort "worst" and can't be the survivor.
    d2 = torch.where(alive, d2, torch.full_like(d2, float("inf")))

    # Sort by (cell, d2). For each cell, keep the lowest-d2 alive Gaussian;
    # mark the rest of the alive Gaussians in that cell for eviction.
    n_cells = nb_h * nb_w
    keys = live_bin.long() * (1 << 32) + (d2 * 1e6).long().clamp(min=0, max=(1 << 32) - 1)
    # Stable argsort: torch.argsort doesn't support stable on MPS reliably;
    # we'll just take it and accept ties broken by index order.
    order = torch.argsort(keys)
    sorted_bins = live_bin[order]
    sorted_alive = alive[order]
    # Mark first occurrence per cell (the keeper).
    is_first = torch.ones_like(sorted_bins, dtype=torch.bool)
    if sorted_bins.numel() > 1:
        is_first[1:] = sorted_bins[1:] != sorted_bins[:-1]
    keep_in_sorted = is_first & sorted_alive
    # Convert back to original index space.
    keepers = torch.zeros_like(alive, dtype=torch.bool)
    keepers[order] = keep_in_sorted
    # Evict everything alive that isn't a keeper.
    evict = alive & ~keepers
    if evict.any():
        canvas.alive = alive & keepers

    # Recompute occupancy using only keepers.
    occupancy = torch.zeros(n_cells, dtype=torch.bool, device=canvas.device)
    occupancy[live_bin[canvas.alive]] = True
    empty_cells = (~occupancy).nonzero(as_tuple=False).flatten()  # cell ids needing spawn

    # All slots that just died (evicted) plus any slots that were already dead.
    dead_idx = (~canvas.alive).nonzero(as_tuple=False).flatten()
    n_spawn = min(int(empty_cells.numel()), int(dead_idx.numel()))
    if n_spawn == 0:
        return 0
    cells = empty_cells[:n_spawn]
    slots = dead_idx[:n_spawn]

    sigma = block * scale * coverage_sigma * 0.5
    canvas.positions[slots] = xy[cells]
    canvas.scales[slots] = float(sigma)
    canvas.rotations[slots] = 0.0
    canvas.colors[slots] = feat[cells]
    canvas.age[slots] = 0
    canvas.error[slots] = 0.0
    canvas.alive[slots] = True
    return n_spawn


# ----------------------------------------------------------- Metrics ----

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def temporal_delta(curr: torch.Tensor, prev: torch.Tensor) -> float:
    """Mean absolute frame-to-frame delta (over all pixels & channels)."""
    return (curr.clamp(0, 1) - prev.clamp(0, 1)).abs().mean().item()


def temporal_delta_flat(
    curr: torch.Tensor, prev: torch.Tensor, motion: torch.Tensor, threshold: float = 0.5
) -> float:
    """Frame-to-frame delta restricted to flat (low-motion) regions.

    motion: (2, H, W). flat = ||motion|| < threshold pixels.
    """
    H, W = curr.shape[-2:]
    # Resize motion to match curr resolution (motion may be at any res).
    if motion.shape[1:] != (H, W):
        motion_up = F.interpolate(
            motion.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
    else:
        motion_up = motion
    mag = motion_up.norm(dim=0)  # (H, W)
    mask = (mag < threshold).float()
    if mask.sum() < 1.0:
        return float("nan")
    diff = (curr.clamp(0, 1) - prev.clamp(0, 1)).abs().mean(dim=0)  # (H, W)
    return ((diff * mask).sum() / mask.sum()).item()


# --------------------------------------------------------- Conditions ----

def run_bicubic(
    lrs: List[torch.Tensor],
    hr_hw: Tuple[int, int],
) -> List[torch.Tensor]:
    out = []
    H, W = hr_hw
    for lr in lrs:
        up = F.interpolate(lr.unsqueeze(0), size=(H, W), mode="bicubic", align_corners=False).squeeze(0)
        out.append(up.clamp(0, 1))
    return out


def _render_normalized(canvas) -> torch.Tensor:
    """Render the canvas with per-pixel weight normalization (alpha blend).

    The reference rasterizer accumulates ``weight * feat`` without dividing
    by the per-pixel total weight, which produces brightness artefacts that
    depend on Gaussian overlap density. This wrapper renders both the
    feature image and a 1-channel "alpha" image (colour := 1) and divides
    so each pixel's output is the weighted *average* of contributing
    Gaussians' features. That is the formula every shipped Gaussian
    rasterizer (gsplat, 3DGS, Image-GS top-k) actually computes; it is the
    fair comparison target for "what does the canvas representation look
    like at this pixel".
    """
    from oss.gaussian.renderer import GaussianBatch
    gb = canvas.snapshot()
    if gb.num_gaussians == 0:
        return torch.zeros(
            (canvas.feat_dim, *canvas.output_hw),
            dtype=canvas.dtype, device=canvas.device,
        )
    feat_aug = torch.cat([gb.feat, torch.ones(gb.num_gaussians, 1, device=gb.feat.device, dtype=gb.feat.dtype)], dim=1)
    gb_aug = GaussianBatch(xy=gb.xy, scale=gb.scale, rot=gb.rot, feat=feat_aug)
    out = canvas._rasterizer(gb_aug, canvas.output_hw)  # (F+1, H, W)
    rgb = out[: canvas.feat_dim]
    a = out[canvas.feat_dim:].clamp(min=1e-4)
    return (rgb / a).clamp(0, 1)


def run_canvas(
    lrs: List[torch.Tensor],
    flows_to_curr: List[torch.Tensor],
    hr_hw: Tuple[int, int],
    scale: int,
    block: int,
    *,
    alpha: float,
    device: torch.device,
    coverage_sigma: float = 1.0,
) -> Tuple[List[torch.Tensor], List[float]]:
    """Run the canvas loop. Returns rendered HR frames + per-frame timing."""
    from oss.gaussian.canvas import PersistentCanvas
    from oss.gaussian.canvas.warp import warp_canvas

    # Capacity: (H_lr/block) * (W_lr/block) plus headroom.
    lr0 = lrs[0]
    lr_h, lr_w = lr0.shape[-2:]
    n_blocks = (lr_h // block) * (lr_w // block)
    capacity = n_blocks  # tight; respawn re-uses dead slots.

    canvas = PersistentCanvas(
        capacity=capacity,
        feat_dim=3,
        output_hw=hr_hw,
        tile_size=16,
        device=device,
        dtype=torch.float32,
    )
    init_canvas_from_lr(canvas, lr0.to(device), scale=scale, block=block, coverage_sigma=coverage_sigma)

    rendered_frames: List[torch.Tensor] = []
    timings: List[float] = []

    # Render first frame (no warp yet).
    t0 = time.time()
    img0 = _render_normalized(canvas)
    if device.type == "mps":
        torch.mps.synchronize()
    timings.append(time.time() - t0)
    rendered_frames.append(img0.cpu())

    for i in range(1, len(lrs)):
        lr = lrs[i].to(device)
        flow = flows_to_curr[i - 1].to(device)  # (2, H, W) at HR — pixel offsets

        t0 = time.time()
        # Apply motion warp (alpha=0 -> no warp; alpha=1 -> full warp).
        warped = warp_canvas(canvas, flow, alpha=float(alpha))
        # Replace canvas state in-place with warped values so the loop is stateful.
        canvas.positions = warped.positions
        canvas.alive = warped.alive
        canvas.age = warped.age + 1  # bump age
        # Re-spawn dead slots from current LR.
        respawn_disoccluded(canvas, lr, scale=scale, block=block, coverage_sigma=coverage_sigma)
        # Render.
        img = _render_normalized(canvas)
        if device.type == "mps":
            torch.mps.synchronize()
        timings.append(time.time() - t0)
        rendered_frames.append(img.cpu())

        # Refresh alive Gaussian colours from current LR via EMA. This is
        # the "temporal accumulation" feature of the canvas — without it,
        # the canvas is just frame-0 forever (alpha=0) or pure frame-by-
        # frame splat (alpha=1, no smoothing). With it, alpha=0 becomes a
        # naive temporal-EMA baseline (no motion compensation), and alpha=1
        # becomes motion-compensated temporal accumulation.
        pos_norm_x = (canvas.positions[:, 0] / hr_hw[1]) * 2.0 - 1.0
        pos_norm_y = (canvas.positions[:, 1] / hr_hw[0]) * 2.0 - 1.0
        grid = torch.stack([pos_norm_x, pos_norm_y], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            lr.unsqueeze(0), grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        ).view(3, -1).t()  # (N, 3)
        # 70% old / 30% new — same EMA for both alpha conditions so the
        # only difference under test is the warp.
        canvas.colors = torch.where(
            canvas.alive.unsqueeze(-1),
            0.7 * canvas.colors + 0.3 * sampled,
            canvas.colors,
        )

    return rendered_frames, timings


# ---------------------------------------------------------------- Main ----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", type=Path, default=Path("data/sintel"))
    ap.add_argument("--sequence", default="alley_1")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--hr-h", type=int, default=432, help="HR height (multiple of tile_size=16)")
    ap.add_argument("--hr-w", type=int, default=512, help="HR width (multiple of tile_size=16)")
    ap.add_argument("--scale", type=int, default=2, help="HR/LR ratio (box downsample)")
    ap.add_argument("--block", type=int, default=4, help="LR block size per Gaussian")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out-dir", type=Path, default=Path("results/naive_canvas_stability"))
    ap.add_argument("--save-frames", action="store_true", default=True)
    ap.add_argument("--coverage-sigma", type=float, default=1.0)
    args = ap.parse_args()

    if args.hr_h % 16 or args.hr_w % 16:
        print(f"ERROR: HR dims must be multiples of 16 (tile_size).", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.manual_seed(0)

    print(f"Loading {args.frames} frames from {args.sequence}, HR={args.hr_h}x{args.hr_w}, "
          f"LR={args.hr_h//args.scale}x{args.hr_w//args.scale} (scale={args.scale}, block={args.block}) ...")
    hrs, lrs, flows = load_sequence(
        args.sintel_root, args.sequence, args.frames, args.hr_h, args.hr_w, args.scale
    )
    print(f"  HR: {hrs[0].shape}, LR: {lrs[0].shape}, motion: {flows[0].shape}, n_motion={len(flows)}")
    n_blocks = (lrs[0].shape[-2] // args.block) * (lrs[0].shape[-1] // args.block)
    print(f"  Naive Gaussian count: {n_blocks} (one per {args.block}x{args.block} LR block)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Running BICUBIC ...")
    bicubic = run_bicubic(lrs, (args.hr_h, args.hr_w))

    print("Running CANVAS alpha=0 (no-warp) ...")
    canv_nowarp, t_nw = run_canvas(
        lrs, flows, (args.hr_h, args.hr_w), args.scale, args.block,
        alpha=0.0, device=device, coverage_sigma=args.coverage_sigma,
    )
    print(f"  mean per-frame time: {sum(t_nw)/len(t_nw):.3f}s")

    print("Running CANVAS alpha=1.0 (full warp) ...")
    canv_warp, t_w = run_canvas(
        lrs, flows, (args.hr_h, args.hr_w), args.scale, args.block,
        alpha=1.0, device=device, coverage_sigma=args.coverage_sigma,
    )
    print(f"  mean per-frame time: {sum(t_w)/len(t_w):.3f}s")

    # ------------------------------------------------------------ metrics
    rows = []
    cumul = {
        "bicubic": {"psnr": [], "delta": [], "delta_flat": []},
        "canvas_nowarp": {"psnr": [], "delta": [], "delta_flat": []},
        "canvas_warp": {"psnr": [], "delta": [], "delta_flat": []},
    }
    for i in range(args.frames):
        gt = hrs[i]
        m_for_flat = flows[i - 1] if i > 0 else None

        for name, frames in [
            ("bicubic", bicubic),
            ("canvas_nowarp", canv_nowarp),
            ("canvas_warp", canv_warp),
        ]:
            cur = frames[i]
            p = psnr(cur, gt)
            cumul[name]["psnr"].append(p)
            if i > 0:
                d = temporal_delta(cur, frames[i - 1])
                df = temporal_delta_flat(cur, frames[i - 1], m_for_flat) if m_for_flat is not None else float("nan")
                cumul[name]["delta"].append(d)
                cumul[name]["delta_flat"].append(df)
            row = dict(
                frame=i,
                cond=name,
                psnr=p,
                delta=cumul[name]["delta"][-1] if i > 0 else float("nan"),
                delta_flat=cumul[name]["delta_flat"][-1] if i > 0 else float("nan"),
            )
            rows.append(row)

    csv_path = args.out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "cond", "psnr", "delta", "delta_flat"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # ------------------------------------------------------------ summary
    summary_lines = []
    summary_lines.append(f"Sequence: {args.sequence}, n_frames={args.frames}, "
                         f"HR={args.hr_h}x{args.hr_w}, LR_scale={args.scale}, block={args.block}")
    summary_lines.append(f"Gaussians: {n_blocks}")
    summary_lines.append("")
    header = f"{'condition':<18} {'PSNR':>8} {'Δ_all':>10} {'Δ_flat':>10}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for name in ("bicubic", "canvas_nowarp", "canvas_warp"):
        psnr_mean = sum(cumul[name]["psnr"]) / len(cumul[name]["psnr"])
        delta_mean = sum(cumul[name]["delta"]) / max(1, len(cumul[name]["delta"]))
        df_vals = [v for v in cumul[name]["delta_flat"] if not (v != v)]  # filter NaN
        delta_flat_mean = sum(df_vals) / max(1, len(df_vals))
        summary_lines.append(f"{name:<18} {psnr_mean:>8.3f} {delta_mean:>10.5f} {delta_flat_mean:>10.5f}")
    summary = "\n".join(summary_lines)
    print()
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n")

    if args.save_frames:
        for name, frames in [
            ("bicubic", bicubic),
            ("canvas_nowarp", canv_nowarp),
            ("canvas_warp", canv_warp),
            ("hr_gt", hrs),
        ]:
            for i, fr in enumerate(frames):
                _save_png(fr, args.out_dir / "frames" / name / f"frame_{i:04d}.png")
        print(f"Wrote frames under {args.out_dir/'frames'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

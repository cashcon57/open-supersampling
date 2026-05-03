"""Measure inference-time VRAM at typical user resolutions.

Loads the latest SR-CNN checkpoint from --output-dir, runs forward at a few
representative resolutions, prints peak VRAM. Inference mode (no_grad +
train(False)). Mirrors the actual deployed inference path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from oss.sr import build_sr_model


def measure(sr_model, device: str, lr_h: int, lr_w: int, scale: int = 2,
            n_warmup: int = 3, n_runs: int = 5) -> dict:
    sr_model.train(False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    x = torch.zeros(1, 12, lr_h, lr_w, device=device)

    # Warmup.
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = sr_model(x)
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(n_runs):
            out = sr_model(x)
        torch.cuda.synchronize(device)
    elapsed = (time.monotonic() - start) / n_runs

    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    out_h, out_w = out.shape[-2:]
    return {
        "lr": (lr_h, lr_w),
        "hr": (out_h, out_w),
        "peak_mib": peak,
        "reserved_mib": reserved,
        "ms_per_frame": elapsed * 1000,
        "fps": 1.0 / elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args()

    ckpts = sorted(args.output_dir.glob("step-*.pt"))
    if not ckpts:
        print(f"FAIL: no checkpoints in {args.output_dir}")
        return 1

    latest = ckpts[-1]
    ck = torch.load(latest, map_location=args.device, weights_only=False)
    saved_args = ck.get("args", {})
    tier = saved_args.get("tier", "lite")
    sr_backbone = saved_args.get("sr_backbone", "simple")

    factory_kind = "rrdb" if (sr_backbone == "rrdb") else "simple"
    sr_model = build_sr_model(
        model_kind=factory_kind, tier=tier, in_channels=12, scale=args.scale
    ).to(args.device)
    sr_model.load_state_dict(ck["sr_model"])

    n_params = sum(p.numel() for p in sr_model.parameters())
    print(f"Checkpoint: {latest.name}")
    print(f"  tier={tier}  backbone={factory_kind}  params={n_params:,}")
    print(f"  weights size = {n_params * 4 / 1024**2:.1f} MiB (FP32)")
    print()

    # Representative resolutions: LR side. Output is 2x.
    cases = [
        ("Steam Deck (1280x800 LR -> 2560x1600)", 800, 1280),
        ("720p LR -> 1440p HR",                   720, 1280),
        ("900p LR -> 1800p HR",                   900, 1600),
        ("1080p LR -> 2160p HR (4K)",            1080, 1920),
    ]

    print(f"{'config':40s}  {'peak VRAM':>10s}  {'reserved':>10s}  {'ms/frame':>10s}  {'fps':>6s}")
    print("-" * 90)
    for label, h, w in cases:
        try:
            r = measure(sr_model, args.device, h, w, scale=args.scale)
            print(f"{label:40s}  {r['peak_mib']:8.1f} MiB  {r['reserved_mib']:8.1f} MiB  "
                  f"{r['ms_per_frame']:8.2f} ms  {r['fps']:5.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{label:40s}  OOM")
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

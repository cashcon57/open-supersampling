"""Benchmark all the inference optimisations side by side.

Loads the latest SR-CNN checkpoint (full or lean) and measures peak VRAM +
ms/frame across:

  - PyTorch FP32 baseline
  - + channels-last
  - + FP16
  - + channels-last + FP16
  - + channels-last + FP16 + CUDA Graphs
  - + lean checkpoint (if --lean-ckpt provided)

Stacks each optimisation on top of the previous so the marginal gain is
readable. Each row uses ``oss.sr.inference.SRInferenceEngine``; warm-up
runs aren't timed; reported peak VRAM is post-warmup.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from oss.sr.inference import SRInferenceEngine


def bench(engine: SRInferenceEngine, lr_h: int, lr_w: int, n_runs: int = 20) -> dict:
    torch.cuda.reset_peak_memory_stats(engine.device)
    torch.cuda.empty_cache()
    engine.warm(lr_h, lr_w, n_warmup=5)

    x = torch.zeros(1, engine.in_channels, lr_h, lr_w, device=engine.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(engine.device)
    torch.cuda.synchronize(engine.device)
    start = time.monotonic()
    for _ in range(n_runs):
        out = engine(x)
    torch.cuda.synchronize(engine.device)
    elapsed = (time.monotonic() - start) / n_runs

    return {
        "lr": (lr_h, lr_w),
        "hr": tuple(out.shape[-2:]),
        "peak_mib": torch.cuda.max_memory_allocated(engine.device) / 1024**2,
        "ms": elapsed * 1000,
        "fps": 1.0 / elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="Full checkpoint (12-channel).")
    p.add_argument("--lean-ckpt", type=Path, default=None, help="Optional lean checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"FAIL: checkpoint not found: {args.ckpt}")
        return 1

    cases = [
        ("Steam Deck (1280x800 LR)", 800, 1280),
        ("720p LR -> 1440p HR",      720, 1280),
        ("1080p LR -> 4K HR",       1080, 1920),
    ]

    configs = [
        ("PyTorch FP32 baseline",          {"fp16": False, "channels_last": False, "cuda_graphs": False}),
        ("+ channels-last",                {"fp16": False, "channels_last": True,  "cuda_graphs": False}),
        ("+ FP16",                         {"fp16": True,  "channels_last": False, "cuda_graphs": False}),
        ("+ FP16 + channels-last",         {"fp16": True,  "channels_last": True,  "cuda_graphs": False}),
        ("+ FP16 + ch-last + CUDA graph",  {"fp16": True,  "channels_last": True,  "cuda_graphs": True}),
    ]

    for label, h, w in cases:
        print(f"\n=== {label} ({h}x{w} LR -> {h*2}x{w*2} HR) ===")
        print(f"{'config':40s}  {'peak VRAM':>10s}  {'ms/frame':>10s}  {'fps':>8s}")
        print("-" * 75)
        baseline_ms = None
        for cfg_label, kwargs in configs:
            try:
                eng = SRInferenceEngine.from_checkpoint(args.ckpt, device=args.device, **kwargs)
                r = bench(eng, h, w)
                speedup = f"{baseline_ms / r['ms']:.2f}x" if baseline_ms else "1.00x"
                if baseline_ms is None:
                    baseline_ms = r["ms"]
                print(f"{cfg_label:40s}  {r['peak_mib']:8.1f} MiB  {r['ms']:8.2f} ms  "
                      f"{r['fps']:6.1f} ({speedup})")
            except Exception as e:
                print(f"{cfg_label:40s}  ERROR: {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    if args.lean_ckpt and args.lean_ckpt.exists():
        print(f"\n=== Lean (9-channel) checkpoint ===")
        for label, h, w in cases:
            try:
                eng = SRInferenceEngine.from_checkpoint(
                    args.lean_ckpt, device=args.device,
                    fp16=True, channels_last=True, cuda_graphs=True,
                )
                r = bench(eng, h, w)
                print(f"  {label:38s}  {r['peak_mib']:8.1f} MiB  {r['ms']:8.2f} ms  "
                      f"{r['fps']:6.1f} fps")
            except Exception as e:
                print(f"  {label:38s}  ERROR: {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

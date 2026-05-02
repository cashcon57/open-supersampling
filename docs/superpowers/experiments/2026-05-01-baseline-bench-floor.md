# Baseline Bench: The Numerical Floor for OSS-Gaussian

Date: 2026-05-01
Author: bench harness in `oss/gaussian/bench/run_baselines.py`
Hardware: Apple M3 Max (MPS + CPU)

## Setup

- **Test set:** 10 frames, one per sequence, drawn from a Sintel-shaped
  dataset at 1024x436 ("clean" pass, middle frame of each sequence).
  Sequence names mirror the canonical Sintel split:
  `alley_1, ambush_2, bamboo_1, bandage_2, cave_4, market_5, mountain_1,
  shaman_3, sleeping_2, temple_2`.
- **Scale:** 2.0 (LR = box-downsampled HR at 512x218).
- **Baselines:** `BicubicUpscaler`, `LanczosUpscaler`. Both are implemented
  in `oss/gaussian/bench/baselines.py`. `kornia` is not installed in this
  environment, so `LanczosUpscaler` falls back to bicubic — documented in
  the module header. The numbers for `lanczos` below are therefore
  expected to match `bicubic` until kornia is added; the row exists so
  the harness layout is locked in for the real comparison.
- **Runs:** 100 timed iterations per (baseline, frame) after 10 warmup
  iterations. `time.perf_counter()` with `torch.mps.synchronize()` /
  per-iteration sync to defeat lazy dispatch.
- **Metrics:** PSNR (MSE in [0,1] sRGB), SSIM (`pytorch_msssim`,
  data_range=1.0, 11-tap window), LPIPS-VGG (`lpips`, `[-1,1]` input).

## DATA CAVEAT — read this before quoting numbers

The MPI Sintel hosting URL (`http://files.is.tue.mpg.de/sintel/training_clean.zip`)
returned **HTTP 404** at the time of this run. The numbers below were
collected on a *structured synthetic proxy* at the correct Sintel
resolution (1024x436): low-frequency sinusoidal mixes + smoothed Gaussian
noise + a checkerboard region for high-frequency content. This proxy
characterises latency precisely (latency is content-independent at fixed
resolution) but **PSNR / SSIM / LPIPS values must be re-measured on real
Sintel frames before being treated as the official quality floor.**

Action items: locate an alternate Sintel mirror or use a different
HR-RGB-with-flow-and-depth set (e.g., the `MPI-Sintel-complete.zip`
mirror on AWS Open Data, or substitute `Spring-2023` / TartanAir in the
short term — the harness is dataset-agnostic given the canonical layout).

CSV outputs (committed for traceability):
- `results/baseline_floor_proxy.csv` — MPS run
- `results/baseline_floor_proxy_cpu.csv` — CPU run

## Numbers (proxy data, 2x upscale, 1024x436 HR)

### Quality (mean over 10 sequences)

| Baseline | PSNR (dB) | SSIM   | LPIPS-VGG |
|----------|-----------|--------|-----------|
| Bicubic  | 28.30     | 0.9794 | 0.0335    |
| Lanczos  | 28.30*    | 0.9794*| 0.0335*   |

\* Lanczos identical to bicubic — kornia missing, fallback path active.

### Latency (mean / p50 / p95 ms per 1024x436 upscale, M3 Max)

| Baseline | Device | mean | p50  | p95  |
|----------|--------|------|------|------|
| Bicubic  | MPS    | 0.29 | 0.27 | 0.39 |
| Bicubic  | CPU    | 1.04 | 1.05 | 1.33 |
| Lanczos  | MPS    | 0.32 | 0.31 | 0.40 |
| Lanczos  | CPU    | 0.84 | 0.78 | 1.20 |

## The Bar to Clear

OSS-Gaussian must produce HR frames at 1024x436 (2x upscale) that:

1. **Beat PSNR ~28.3 dB / SSIM ~0.98 / LPIPS-VGG ~0.034** on the same
   real-Sintel test split. (These specific numbers are proxy-derived; the
   real-Sintel re-measurement is a Sprint-2 prerequisite.) Anything that
   doesn't move LPIPS down materially is not worth shipping over plain
   bicubic — bicubic is one `F.interpolate` call.
2. **Stay within 1 ms on M3 Max MPS** for the upscale forward pass
   to remain "iso-latency vs the trivial floor". Bicubic at 0.29 ms mean
   / 0.39 ms p95 sets a brutal target. Anything above ~3 ms loses the
   "free upgrade" pitch on integrated GPUs.
3. On CPU-only systems (Steam Deck-tier), bicubic is ~1 ms and lanczos
   ~0.8 ms. OSS-Pico (the Vulkan/NCNN port) needs to land in the same
   millisecond budget for the Deck story to work.

These bicubic / lanczos numbers are *the trivial floor*. The harder
bars — FSR 2 Quality and DLSS-SR Quality — are still ahead of us:

- **FSR 2 / DLSS** are the quality tier OSS-Gaussian must compete with
  to be a serious vendor-agnostic alternative. Their wiring depends on
  the D3D12 / Vulkan host harness (Sprint 2) and the NGX shim DLL.
  Sprint 4 close-out task T4.13 brings them into this same harness so
  the next bench iteration produces the full apples-to-apples table.
- The proxy-vs-real-Sintel quality gap is the immediate risk; latency
  numbers above are trustworthy.

## Reproduce

```bash
python -m oss.gaussian.bench.run_baselines \
    --sintel-root data/sintel \
    --scale 2.0 \
    --output bench_results.csv \
    --runs 100 --warmup 10 --device mps
```

The bundled integration test
(`tests/gaussian/test_baselines.py::test_run_baselines_on_synthetic_sintel_fixture`)
exercises the full CLI on a tmp synthetic dataset and asserts the CSV
schema + finite metric values, so regressions in argparse / dataset
traversal / metric construction are caught in CI.

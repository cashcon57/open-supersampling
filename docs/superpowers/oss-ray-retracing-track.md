# OSS Ray-Retracing — Denoising / DLSS-RR Replacement Track

**Ray-Retracing** — OSS's temporal denoising + spatial reconstruction component. We don't cast new rays; we reuse existing samples by reprojecting them via motion vectors — tracing the original camera ray's screen-space path backward through time. Same surface area as DLSS Ray Reconstruction; different algorithm (we use the persistent Gaussian canvas as the temporal accumulator rather than a learned denoiser network).

**Date created:** 2026-05-02
**Pivot trigger:** `docs/superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md`
**Status:** scaffolding only — production work blocked on NoiseBase data.

## Why this track exists

The Gaussian splat representation **cannot do** single-image super-resolution. It **can do** denoising — `docs/superpowers/experiments/2026-05-01-gaussian-denoising-naive-test.md` (D1) showed Image-GS at n=1000 beating OIDN on PSNR (26.90 vs 26.56 dB) and beating Gaussian blur on PSNR + LPIPS for 5/6 frames. The Gaussian prior is sound for denoising; it just can't hallucinate the high-frequency detail that SR demands.

DLSS Ray Reconstruction (NVIDIA's released-2023 successor to DLSS-DR + ML denoising) is the natural product target. NRD is the open baseline. OSS Ray-Retracing is the open vector-domain alternative.

## Architectural sketch (v0, deferred until data lands)

```
LR colour (B, 3, H, W)              ┐
viewZ / depth (B, 1, H, W)          │
normals (B, 3, H, W)                ├─► encoder (small UNet)
roughness (B, 1, H, W)              │
motion vectors (B, 2, H, W)         │
prev frame canvas hint (B, 3, H, W) │
                                    ▼
                                  param net
                                    │ predicts per-tile (Δposition, bank weights, color, opacity)
                                    ▼
                                 OutputHead → GaussianBatch
                                    │
                                  Rasterizer (sum / alpha-composite TBD)
                                    │
                                    ▼
                              Denoised colour HR
```

The same Sprint 4 modules (`oss/gaussian/network/{prior_bank,param_net,output_head}`, `oss/gaussian/renderer/`) carry over verbatim. The G-buffer interface gets one more channel (roughness) and a different loss target (clean GT instead of HR upsample target).

## Prerequisites

1. **NoiseBase complete download** on the 3080 Ti. Currently only `.zip.part` partials present per the validation memo. ~hundreds of GB; needs a sustained download or a clean restart.
2. **D1 re-run on real (not synthetic) NoiseBase HDR frames.** The 2026-05-01 D1 result used synthetic Monte-Carlo noise; the real data lifts the signal-to-noise burden but adds a HDR exposure dimension we haven't handled yet.
3. **Pick a renderer compositing mode** — current `rasterize_gaussians_sum` is fine for SR-color but for denoising with overlapping geometry we may want alpha-OVER. Test on a small case before committing.
4. **Loss design** — denoising metrics (PSNR HDR, SSIM, LPIPS, possibly an FFT-based metric for noise residue) differ from SR metrics. Plumbing for HDR-aware L1 (Reinhard tonemap → L1) is in the spec but not yet wired.

## What's blocked by this

- Sprint 5 (persistent canvas) revives in this context — temporal accumulation of per-frame Gaussians is exactly what Ray-Retracing wants.
- Sprint 6 (frame extrapolation) is similarly relevant — extrapolation needs a usable per-frame denoised buffer.

## What's NOT blocked

OSS-SR ships independently as a CNN-based pipeline. Ray-Retracing work happens in parallel (or sequentially after SR ships) without delaying the SR product.

## Open questions

1. Does the Gaussian prior survive the move from synthetic noise (D1) to real Monte-Carlo path-traced HDR? Likely yes per the literature, but needs the data.
2. Do alpha-OVER vs sum compositing produce materially different denoising quality? GSASR-style results suggest both can work; needs an A/B on a controlled scene.
3. Can the same network from Sprint 4 (after the dead-init fixes from `2026-05-02-output-head-dead-init.md`) be repurposed for Ray-Retracing by changing the input channels and loss, or do we want a fresh architecture? Almost certainly the former — the Sprint 4 module is tier-scalable and well-tested.

## Status flags

- ⏳ **NoiseBase download** (background task on 3080 Ti, currently incomplete).
- 🔒 **Production training** (blocked on data).
- ✅ **Architecture skeleton ready** (Sprint 4 modules carry over).
- ✅ **D1 result on synthetic noise** (positive, shows representation is sound for denoising).

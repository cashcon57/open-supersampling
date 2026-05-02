# OSS-Gaussian — V0 Paper Draft (Workshop-Shaped, SKELETAL)

> **Status:** draft skeleton, NOT submission-ready. Filled out as experiment memos in `docs/superpowers/experiments/` produce results. Do not promote to a full draft until V0 (single-frame Gaussian SR) clears its bicubic-baseline gate per `gaussian-network-architecture.md` §7.

**Working title:** *Engine-Aliased Training for Real-Time 2D Gaussian Super-Resolution in Game Pipelines*

**Target venue (if results land):** CVPR-W or ICCV-W (workshop-tier engineering note). Move up only if results materially exceed FSR2/DLSS Quality.

---

## 1. Abstract

> Fill last. One paragraph: problem, approach, headline result, repo URL.

---

## 2. Introduction

- **Problem.** Real-time game upscaling is dominated by closed-source pixel-domain CNNs (DLSS, FSR3, XeSS) that ship with hardware. Open vector-domain alternatives are rare. Vector-based representations promise structurally ghost-free temporal accumulation and free frame extrapolation, but pure 2D Gaussians blur high-frequency edges \cite{imagegs2024,gaussianbillboards2024,contour2dgs2025}.
- **Contribution claims (TENTATIVE — replace with measured wins):**
  1. Engine-aliased LR synthesis pipeline (Halton jitter + TAA blur + JPEG) that avoids the bicubic-LR-trap \cite{realesrgan2021}.
  2. Anisotropic G-buffer-conditioned Gaussian param network (per-tile (normal, depth-grad) bias on bank softmax).
  3. Open MIT-licensed implementation with vendored gsplat \cite{gsasr2025} + reference PyTorch fallback.
  4. Quantitative comparison vs bicubic / FSR2 Quality / Real-ESRGAN on Sintel / SRGD / Cyberpunk-capture data.
- Anchor in literature: \cite{gaussiansr2024,gsasr2025,gs_stvsr2026}.

## 3. Related Work

- **Pixel-domain SR:** ESRGAN family, Real-ESRGAN \cite{realesrgan2021}, SwinIR.
- **2D Gaussian SR:** GaussianSR \cite{gaussiansr2024}, GSASR \cite{gsasr2025}, GS-STVSR \cite{gs_stvsr2026}.
- **Anisotropic / texture-aware splats:** Image-GS \cite{imagegs2024}, Gaussian Billboards \cite{gaussianbillboards2024}, Contour-aware 2DGS \cite{contour2dgs2025}.
- **Industrial baselines:** DLSS, FSR3, XeSS — closed but documented via Streamline \cite{nvidia_streamline}.

## 4. Method

> Fill from `gaussian-network-architecture.md`. Sections 4.1–4.4 below mirror that doc's §1–§4.

### 4.1 Tile-aware Gaussian param network

- 4-level UNet at LR resolution.
- Output: per-tile (Δposition, K bank-weight logits, color) for K Gaussians per complex tile.
- Tier scaling: pico/lite/standard/ultra knobs on channel widths and K. **Pico is distillation-only post-MVP** (smoke results: pico 75K params can not learn SR from scratch at 540×960; \cite{[smoke-findings-2026-05-02]}).

### 4.2 Covariance Prior Bank

- Fixed 16-entry vocabulary of (sx, sy, θ).
- Network predicts a softmax over the 16; final per-Gaussian (sx, sy, θ) is the weighted geometric/circular mean.

### 4.3 Anisotropic G-buffer-conditioned bias

- Per-tile (mean normal, mean depth gradient) → 5-feature linear projection → additive bias on bank logits before softmax.
- Zero-init: enabling the flag is graceful (matches disabled until trained).

### 4.4 Engine-aliased LR synthesis

- Halton(2,3) subpixel jitter (idx+1, matching Unreal/DLSS convention).
- Area-filter downsample.
- Configurable Gaussian TAA blur (σ=0.5 mild → σ=1.5 aggressive).
- Optional JPEG q≥85 for content-delivery scenarios.

## 5. Experiments

> Pull tables straight from `docs/superpowers/experiments/*.md`.

### 5.1 Datasets

- Sintel (clean + flow; depth supplement TBD).
- SRGD (GameEngineData ↔ DownscaleData paired).
- HyperSim (deferred).
- Cyberpunk capture (gated on V0 success).

### 5.2 Baselines

- Bicubic upsample (`F.interpolate(antialias=True)`).
- Lanczos (Sprint 1 baseline implementation).
- FSR2 Quality (on hardware; iso-latency comparison per Sprint 4 close-out gate).
- Real-ESRGAN (off-the-shelf weights).

### 5.3 Headline results

> **Pending V0 gate clear.** Fill from experiment memos.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (ms) |
|---|---|---|---|---|
| Bicubic | TBD | TBD | TBD | TBD |
| Lanczos | TBD | TBD | TBD | TBD |
| FSR2 Quality | TBD | TBD | TBD | TBD |
| Real-ESRGAN | TBD | TBD | TBD | TBD |
| **Ours (V0)** | TBD | TBD | TBD | TBD |
| Ours (V0.5 + pixel residual) | TBD | TBD | TBD | TBD |

### 5.4 Ablations

- Bank size 8 vs 16 vs 32 (T4.9, deferred to v1 if v0 ships).
- K Gaussians per tile (3 vs 5 vs 8).
- G-buffer bias on/off.
- Engine-aliased LR severity (σ=0.5 vs 1.5).
- Pure Gaussian vs + pixel-residual head.

## 6. Discussion / Limitations

- Pico tier underperforms; we ship distillation-only.
- Bicubic-LR-trap on standard benchmarks; real comparison requires engine-aliased LR.
- Frame extrapolation is **not** "free" — disocclusion needs learned repair (deferred to V1.5).
- 3080 Ti single-GPU only; multi-day training; no DDP. Result generalization to longer training runs untested.

## 7. Conclusion

> Fill last.

## 8. Reproducibility

- All code MIT-licensed at `https://github.com/cashcon57/open-supersampling`.
- Each table cell traces back to a commit SHA and an experiment memo path.
- Run `scripts/<train-host>-sprint4-smoke.ps1` to reproduce smoke-test signal.

## Acknowledgements / References

> Bibliography via `oss-gaussian.bib` once compiled.

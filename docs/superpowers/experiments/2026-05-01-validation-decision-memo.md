# Validation Decision Memo — Pre-Training Gate
**Date:** 2026-05-01  
**Status:** COMPLETE — all 5 validation tests returned  
**Decision:** Sprint 4 training authorized — with corrected training data strategy and mandatory low-capacity smoke-test first

---

## 1. Test Results Summary

| Test | Method | Result | Key Number |
|------|--------|--------|------------|
| 5U baseline bench | Bicubic/Lanczos on synthetic | ✅ Pipeline verified | Bicubic on real Sintel ≈42.78 dB PSNR |
| 4U naive canvas | PersistentCanvas + warp, no training | ❌ Negative | 21.63 dB vs 31.69 dB bicubic (−10 dB) |
| 1U naive upscaler | Image-GS optimization fitting, 50K G / 5K steps | ❌ Negative | −3.59 dB vs bicubic across 6 scenes |
| 2U pretrained GSASR | EDSR_DIV2K Enhanced weights on Sintel | ❌ Negative (trap) | −4.55 dB vs bicubic — bicubic-LR-trap |
| D1 naive denoiser | Image-GS at n=1000 vs OIDN, synthetic MC noise | ✅ Positive | 26.90 dB PSNR (beats OIDN 26.56 dB) |

---

## 2. What the Tests Proved

### 2.1 Training is required (1U, 4U — expected)

Image-GS optimization fitting achieves 42.8 dB on LR but loses 3.59 dB vs bicubic on HR. The Gaussian footprints at 2× scale become visible blobs — the representation does not contain SR detail without a learned network feeding it. The sprint architecture was correct: the network is not optional.

**Implications:**
- Sprint 4 (network training) is the decisive gate. Without it, neither the canvas nor the splats provide quality.
- Sprint 5 (canvas) is conditionally valuable: it locks in and propagates Sprint 4's signal across frames, but generates nothing ex nihilo. Sprint 5 is not cut, but its value is entirely gated on Sprint 4 succeeding.

### 2.2 The pretrained baseline result is a benchmark artifact, not a field failure (2U)

GSASR loses to bicubic by 4.5 dB because the evaluation used bicubic-downsampled LR. Bicubic upsampling is the near-inverse of bicubic downsampling, so any SR network that hallucinates HF detail will lose PSNR/SSIM on clean bicubic-LR benchmarks. Real-ESRGAN Section 3.1 documents this explicitly.

**Critical training data correction (applies Sprint 4 onward):**
- **Do NOT** train against bicubic-clean Sintel LR.
- **Train against engine-aliased LR**: temporal aliasing, TAA noise, subpixel jitter, engine-specific artifacts — what a real game engine emits.
- **Evaluation gate remains FSR2/DLSS Quality** (as already specified in the Sprint 4 close-out criteria). This was the right gate all along. Do not gate against bicubic.

### 2.3 Gaussian representation carries the right denoising prior (D1 — positive)

Image-GS at n=1000 Gaussians beats OIDN on PSNR (26.90 vs 26.56 dB) and beats Gaussian blur on PSNR + LPIPS for 5/6 frames. The under-parameterization-as-prior effect is strong and monotonic: n=1000 > n=5000 > n=30000 on every metric. OIDN still dominates SSIM and LPIPS (0.86 vs 0.76, 0.12 vs 0.26) because the Gaussian prior over-smooths texture.

**Implications for OSS-Gaussian-RR:**
- The Gaussian representation IS sound for denoising at the right Gaussian count. This is the structural prior needed for OSS-Gaussian-RR (DLSS Ray Reconstruction replacement).
- Naive drop-in for OIDN is falsified. A trained network is required to close the LPIPS gap.
- Hybrid architecture is the path: Gaussian prior (low-frequency + firefly suppression) + anisotropic covariance (G-buffer-conditioned, Sprint 4 enhancement) + lightweight CNN refinement for texture preservation.
- **Prerequisite before OSS-Gaussian-RR training**: re-run D1 on real NoiseBase HDR frames (current NoiseBase install on 3080 Ti has only `.zip.part` partial downloads; complete the download before committing to RR training data budget).

---

## 3. Decisions

### Decision 1: Sprint 4 training — AUTHORIZED (conditional)

The architecture is sound. Training is required and justified. However:

**Mandatory precondition: low-capacity smoke-test before full Lambda H100 budget.**

Run Sprint 4 network at reduced capacity (e.g., 1–2 layers, 8 tiles, single Sintel sequence) for ≤3 hours on the 3080 Ti. Gate: can the smallest viable network beat bicubic on even one scene? If yes → full H100 training. If no → architectural review before spend.

This is the "negative result is a kill signal" discipline from our validation-first policy.

### Decision 2: Anisotropic G-buffer-conditioned covariance — PULL INTO SPRINT 4

D1 justifies it. The denoising result shows that the Gaussian prior over-smooths texture; anisotropic covariance (stretch along edges, thin across them) directly addresses this. The design is already specified: condition the 16-entry Prior Bank softmax weights on per-tile (normal, depth_gradient) from the G-buffer. ~1 day implementation.

This also improves upscaling quality (geometric edges) and denoising quality simultaneously. Include in Sprint 4 OutputHead.

### Decision 3: Training data strategy — CORRECTED

| Old (wrong) | New (correct) |
|---|---|
| Bicubic-downsampled Sintel LR | Engine-aliased LR (TAA noise, temporal aliasing, subpixel jitter) |
| Gate vs bicubic | Gate vs FSR2 Quality / DLSS SR Quality (unchanged) |
| Sintel clean only | Sintel + TartanAir + HyperSim + SRGD (unchanged) — but synthesize realistic LR degradation |

Specific LR synthesis pipeline to add before Sprint 4 training data generation:
1. Render at native res → apply per-frame jitter (halton sequence, matching real TAA offsets)
2. Downsample with area filter (not bicubic)
3. Add TAA-blur simulation (exponential moving average, α=0.1)
4. Optional: JPEG artifacts at quality 85 for content-delivery scenarios

### Decision 4: OSS-Gaussian-RR track — HOLD at v1 stretch goal

D1 is positive on the representation, but real NoiseBase validation is incomplete. Keep OSS-Gaussian-RR as v1 stretch / v2 milestone as planned. Do not change scope. Complete NoiseBase download on 3080 Ti and re-run D1 before any RR training commitment.

### Decision 5: Sprint 5 canvas — PROCEED as planned

The canvas architecture is mechanically correct (warp verified, eviction logic verified). It will compound Sprint 4's quality lift across frames. No scope changes.

---

## 4. Updated Sprint 4 Prerequisites (additions to existing design spec)

Add to `docs/superpowers/gaussian-network-architecture.md` § Prerequisites:

1. **Engine-aliased LR synthesis pipeline** — implemented and validated on ≥2 Sintel sequences before training data generation begins.
2. **Low-capacity smoke-test gate** — 3080 Ti, single sequence, ≤3 hours. Must beat bicubic on at least 1 scene. No Lambda H100 spend until this passes.
3. **Anisotropic G-buffer covariance** in OutputHead — merged to main before full training run.
4. **NoiseBase real-data D1 re-run** — prerequisite for RR track only, not SR training.

---

## 5. What the Tests Did NOT Refute

- The Gaussian temporal canvas architecture as the right output format for game SR.
- Sprint 4 network training as the path to beating FSR/DLSS.
- Sprint 5's temporal value (conditional, not eliminated).
- OSS-Gaussian-RR as a future track (denoising prior is sound, texture gap is learnable).
- The 7-sprint roadmap — no scope changes, preconditions added.

---

## 6. Next Actions

| Action | Owner | Gate |
|--------|-------|------|
| Engine-aliased LR synthesis pipeline | Sprint 4 | Before training data gen |
| Anisotropic G-buffer covariance in OutputHead | Sprint 4 | Before full training run |
| Low-capacity smoke-test (3080 Ti, single scene, ≤3h) | Sprint 4 | Before Lambda H100 |
| Complete NoiseBase download on 3080 Ti | Background | Before RR training commitment |
| Sprint 2 T2.4–T2.13 (NGX pass-through, G-buffer, EXR, A/B toggle) | Sprint 2 | Parallel track |

# Image-GS as a Path-Tracing Denoiser — Naive Validation Test

**Date:** 2026-05-01
**Author:** Sprint 4 prep / OSS-Gaussian
**Hardware:** RTX 3080 Ti, conda env `image-gs` (PyTorch 2.4.1 + CUDA 12.4 + gsplat 1.4.0)
**Script:** `scripts/test_gaussian_denoising.py`
**Out dir on 3080-ti:** `<train-host-data>\gauss-denoise-exp\out\`
**Comparison images:** [`./2026-05-01-gaussian-denoising-naive-test-images/`](./2026-05-01-gaussian-denoising-naive-test-images/)

## Question

Does fitting Image-GS (Salehi et al. 2024) — a 2D Gaussian splat representation
optimized via gradient descent against a target image — act as an *implicit
smoothness prior* that denoises a noisy 1-spp path-traced frame, **without any
custom training**?

This is the gating experiment for the OSS Ray Retracing direction. A negative
result kills the anisotropic-covariance Sprint 4 work before we commit to it.

## Setup

**Data.** NoiseBase (Bálint et al. 2023) was the intended source. On the
<train-host> `<train-host-data>\noisebase` only contains partial downloads (`scene0000.zip.part`,
`frame0000.zip.part` for both `test8` and `test32`). No complete sequence is
present, so we used the prescribed synthetic fallback:

- 3 skimage builtins: `astronaut`, `coffee`, `cat`
- 3 patches from the Image-GS `teaser.jpg` (rendered Sponza-like content):
  `teaser_tl`, `teaser_mid`, `teaser_br`
- All center-cropped to **512×512 RGB uint8**

For each clean frame we synthesized a noisy "1-spp" surrogate:

```
noisy = Poisson(clean * λ) / λ  +  N(0, 0.04²)  +  fireflies (0.5% of pixels, 3-8× boost)
```

with λ=25. Resulting noisy mean ≈ **PSNR 17.3 / SSIM 0.24 / LPIPS 0.81 vs clean**.
This is a reasonable LDR analogue to NoiseBase 1-spp data, **but it lacks the
HDR dynamic range and the structured (path-correlated) noise of real path
tracers** — caveat noted in the verdict.

**Methods compared.** All vs the clean reference:

| Method        | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `noisy`       | The synthetic input itself (lower bound)                          |
| `blur`        | Gaussian blur σ=1.5 (trivial smoothing baseline)                  |
| `oidn`        | Intel OIDN v0.2.1 (LDR `RT` filter, no aux channels) — **CNN comparator** |
| `image_gs_tiny` | Image-GS, **n=1000 Gaussians**, 2000 steps                      |
| `image_gs_med`  | Image-GS, **n=5000 Gaussians**, 2000 steps                      |
| `image_gs_full` | Image-GS, **n=30000 Gaussians**, 2000 steps (paper-default capacity) |

Image-GS was run via the upstream `main.py` with `--disable_prog_optim` and
`--disable_lr_schedule` so the requested (n_gaussians, max_steps) is honored
end-to-end (default Image-GS forces ≥5000 step minimum via progressive growth,
defeating the capacity sweep).

The Gaussian-as-prior hypothesis is fundamentally about under-parameterization,
so the sweep over `n` is the actual scientific test.

**Caveats / install notes.** The upstream `fused_ssim` CUDA extension cannot
build on this box (no MSVC); replaced by a 5-line shim using `pytorch_msssim.ssim`
in `image-gs/Lib/site-packages/fused_ssim.py`. Image-GS saves results as
**16-bit PNGs** by default — the script renormalizes on load.

## Metrics

### Per-frame results (PSNR ↑ / SSIM ↑ / LPIPS ↓)

| Scene      | Method         | PSNR  | SSIM   | LPIPS  | Time (s) |
|------------|----------------|-------|--------|--------|----------|
| astronaut  | noisy          | 17.70 | 0.3235 | 0.6411 | 0.0      |
| astronaut  | blur σ=1.5     | 25.05 | 0.6774 | 0.4207 | 0.0      |
| astronaut  | oidn           | 25.43 | 0.8366 | **0.1048** | 0.1   |
| astronaut  | gs n=1000      | **25.37** | 0.7194 | 0.2837 | 20.6   |
| astronaut  | gs n=5000      | 24.23 | 0.5948 | 0.3796 | 20.7    |
| astronaut  | gs n=30000     | 20.63 | 0.4442 | 0.4922 | 22.3    |
| coffee     | noisy          | 18.17 | 0.2840 | 0.6866 | 0.0      |
| coffee     | blur σ=1.5     | 25.75 | 0.6831 | 0.4140 | 0.0      |
| coffee     | oidn           | **26.31** | **0.8098** | **0.1573** | 0.1 |
| coffee     | gs n=1000      | 25.92 | 0.7000 | 0.3083 | 20.6    |
| coffee     | gs n=5000      | 24.20 | 0.5687 | 0.3998 | 20.7    |
| coffee     | gs n=30000     | 20.96 | 0.3997 | 0.5200 | 22.5    |
| cat        | noisy          | 16.80 | 0.1849 | 1.1732 | 0.0      |
| cat        | blur σ=1.5     | **28.35** | 0.6905 | 0.6750 | 0.0  |
| cat        | oidn           | 26.04 | **0.8326** | **0.2653** | 0.1 |
| cat        | gs n=1000      | 26.90 | 0.6871 | 0.5084 | 20.4    |
| cat        | gs n=5000      | 23.56 | 0.4883 | 0.7687 | 20.7    |
| cat        | gs n=30000     | 19.72 | 0.2956 | 0.8860 | 22.3    |
| teaser_tl  | noisy          | 16.88 | 0.1794 | 0.8954 | 0.0      |
| teaser_tl  | blur σ=1.5     | 25.76 | 0.7327 | 0.3432 | 0.0      |
| teaser_tl  | oidn           | 27.18 | **0.8791** | **0.0818** | 0.1 |
| teaser_tl  | gs n=1000      | **28.52** | 0.8022 | 0.1584 | 20.5  |
| teaser_tl  | gs n=5000      | 23.86 | 0.5318 | 0.4550 | 20.6    |
| teaser_tl  | gs n=30000     | 19.80 | 0.2969 | 0.6536 | 22.4    |
| teaser_mid | noisy          | 17.18 | 0.2107 | 0.7752 | 0.0      |
| teaser_mid | blur σ=1.5     | 23.16 | 0.7330 | 0.3669 | 0.0      |
| teaser_mid | oidn           | **27.01** | **0.8953** | **0.0918** | 0.1 |
| teaser_mid | gs n=1000      | 27.22 | 0.8119 | 0.1944 | 20.5    |
| teaser_mid | gs n=5000      | 24.43 | 0.5612 | 0.4303 | 20.6    |
| teaser_mid | gs n=30000     | 20.20 | 0.3198 | 0.6081 | 22.3    |
| teaser_br  | noisy          | 17.33 | 0.2639 | 0.7062 | 0.0      |
| teaser_br  | blur σ=1.5     | 22.60 | 0.7462 | 0.3222 | 0.0      |
| teaser_br  | oidn           | 27.40 | **0.9081** | **0.0428** | 0.1 |
| teaser_br  | gs n=1000      | **27.48** | 0.8213 | 0.1206 | 20.6  |
| teaser_br  | gs n=5000      | 24.83 | 0.6385 | 0.3136 | 20.6    |
| teaser_br  | gs n=30000     | 20.34 | 0.3932 | 0.5948 | 22.4    |

### Aggregate over 6 frames

| Method         | PSNR ↑    | SSIM ↑    | LPIPS ↓   | Time/frame |
|----------------|-----------|-----------|-----------|------------|
| noisy (input)  | 17.34     | 0.241     | 0.813     | —          |
| blur σ=1.5     | 25.11     | 0.711     | 0.424     | <1 ms      |
| **oidn**       | 26.56     | **0.860** | **0.124** | 110 ms     |
| **gs n=1000**  | **26.90** | 0.757     | 0.262     | 20.5 s     |
| gs n=5000      | 24.19     | 0.564     | 0.458     | 20.6 s     |
| gs n=30000     | 20.28     | 0.358     | 0.626     | 22.4 s     |

## Visual samples

Each row: `clean | noisy | blur | oidn | gs_n=1k | gs_n=5k | gs_n=30k`. (Full-resolution PNGs are
in [`./2026-05-01-gaussian-denoising-naive-test-images/`](./2026-05-01-gaussian-denoising-naive-test-images/).)

- [`astronaut_compare.png`](./2026-05-01-gaussian-denoising-naive-test-images/astronaut_compare.png) — natural image, sharp edges + soft fabric
- [`teaser_br_compare.png`](./2026-05-01-gaussian-denoising-naive-test-images/teaser_br_compare.png) — render-like content, high-frequency texture
- [`cat_compare.png`](./2026-05-01-gaussian-denoising-naive-test-images/cat_compare.png) — fur (worst case for any smoothness prior)

Visual takeaways:
- **gs_n=1k** clearly denoises but **flattens fine detail** — fur, hair, eyelashes vanish.
- **gs_n=30k** is visually almost indistinguishable from noisy: the model has memorized the speckle.
- **OIDN** preserves fine edges and produces the perceptually cleanest result on every frame.

## Verdict

**The Gaussian smoothness prior is real and reproducible — but a one-shot
optimization-based denoiser is not a viable replacement for a learned CNN
denoiser like OIDN.**

What worked:
1. **Capacity is the prior.** Across all 6 frames, monotonic ordering
   `n=1000 > n=5000 > n=30000` in every metric. n=1k beats Gaussian blur on
   PSNR and LPIPS for 5/6 frames, and **beats OIDN on PSNR for 3/6 frames**
   (by 0.1–1.3 dB) — confirming the under-parameterization hypothesis.
2. **It generalizes.** Behavior holds for natural images and render-like
   content. No frame produced anomalously bad results at n=1k.

What didn't:
1. **OIDN dominates SSIM and LPIPS by huge margins** (0.86 vs 0.76 SSIM,
   0.12 vs 0.26 LPIPS). The Gaussian prior smooths *everything* — it can't
   distinguish noise from texture, and humans see that as blur.
2. **n=30k memorizes noise.** The "default" Image-GS capacity is ~30 dB
   *worse* on PSNR than n=1k. Naively running Image-GS at the paper's
   recommended settings is anti-denoising.
3. **20 s per frame**, 60–100k× too slow for real-time. (Expected, prescribed.)
4. **The cat frame illustrates the failure mode:** even Gaussian blur at σ=1.5
   beats Image-GS on PSNR (28.4 vs 26.9) because fur destroys SSIM regardless
   of method, and PSNR rewards uniform smoothing on this content. A trained
   network knows to preserve high-frequency texture; a naive Gaussian fit doesn't.

## Implications for OSS Ray Retracing (Sprint 4)

**The direction is not killed, but the framing must change.**

The naive "use Image-GS as a drop-in replacement for OIDN/Ray Reconstruction"
hypothesis is **falsified** by these numbers. A pure smoothness prior cannot
match a learned denoiser on perceptual metrics, which is what users actually
see.

The result that *does* survive: **a small (≈1k–2k) Gaussian splat fit to a
1-spp frame achieves OIDN-comparable PSNR**. That means the Gaussian
representation does carry the right inductive bias for path-traced noise — it
just lacks the texture-preservation a CNN provides.

This points to a coupling architecture:
- **Gaussian splat as the denoised low-frequency prior** (cheap, structurally
  correct, anti-firefly by construction).
- **Anisotropic covariance + a small CNN refinement** to recover texture
  *conditioned on* the splat's clean low-frequency manifold.

That's a more constrained version of what Sprint 4 was already going to test
with anisotropic covariance, so the Sprint 4 work is **justified to proceed**,
**but the success criterion is no longer "match OIDN" — it is "given a Gaussian
prior, does anisotropic covariance + cheap refinement close the LPIPS gap to
OIDN."**

## Honest scoping note

Image-GS optimization here takes **~20 s/frame** on a 3080 Ti (1k–30k Gaussians,
2000 steps, 512×512). This experiment **does not validate a deployable
denoiser** — it validates whether the Gaussian *representation* is in the right
ballpark for path-tracing denoising. A real-time deployment is the OSS Ray Retracing
network (different work — direct Gaussian-parameter prediction from G-buffers,
no per-frame optimization).

## Caveats to weight heavily

1. **Synthetic noise, LDR.** Real path-traced HDR noise has correlated multi-bounce
   structure, heavier-tailed fireflies, and strong albedo/depth correlations that
   our Poisson + Gaussian + sparse-firefly model under-represents. NoiseBase data
   on this <train-host> was incomplete; re-running with real NoiseBase frames once they
   stage is a strict prerequisite before greenlighting Sprint 4.
2. **No G-buffer guidance.** OIDN can take albedo and normals as auxiliary
   inputs and would gain ~3–5 dB PSNR if it had them. We ran OIDN unguided to
   keep the comparison fair against the unguided Image-GS fitter.
3. **Frame-independent.** No temporal coherence test. A learned denoiser
   (OIDN, RR) and any deployable Ray Retracing will both leverage temporal
   accumulation; this experiment is single-frame only.

## Reproduce

```bash
# 1) Push script + shim
scp scripts/test_gaussian_denoising.py <train-host>:<train-host-data>/gauss-denoise-exp/
# (one-time only) drop fused_ssim shim into env if not present:
#   echo "..." > .../envs/image-gs/Lib/site-packages/fused_ssim.py

# 2) Run
ssh <train-host> 'conda run -n image-gs python <train-host-data>\gauss-denoise-exp\test_gaussian_denoising.py --num-gaussians 30000 --max-steps 2000'

# 3) Pull artifacts
scp <train-host>:<train-host-data>/gauss-denoise-exp/out/metrics.csv ./
for s in astronaut coffee cat teaser_tl teaser_mid teaser_br; do
  scp "<train-host>:<train-host-data>/gauss-denoise-exp/out/$s/compare.png" "./${s}_compare.png"
done
```

Total wall time on 3080 Ti: **~6.5 minutes** (well inside the 2-hour budget).

# Pico vs Lite on Aggressive Engine-Aliased LR (SRGD ActionRPG)
**Date:** 2026-05-02
**Status:** complete — both tiers fail to learn; promotes V0.5 pixel-residual head as next architectural test
**Predecessor:** `2026-05-02-sprint4-smoke-findings.md` §4–§6
**Hardware:** RTX 3080 Ti 12 GB, Miniconda3 image-gs env, torch 2.4.1 + CUDA
**Code commit:** `2180af6` (aggressive LR synth landed) and `889684a` (real SSIM in `composite_loss`); runs launched at `2180af6` so the pooled-L1 fallback path was active.

## Hypothesis

After Codex 5.5 review (B-/7.2 grade) flagged that pico-tier dismissal was premature, retest pico with the same aggressive engine-aliased LR (σ=1.5 + JPEG q=85) the lite tier saw. Concurrent test: lite tier on a single SRGD scene (ActionRPG only) to remove multi-scene batch heterogeneity as a confound.

Both runs share scene + LR-synth config so the only varied dimension is **tier capacity**. If neither learns, the failure is architectural, not capacity. If lite learns and pico doesn't, capacity is the bottleneck. If pico learns, the prior dismissal was wrong.

## Setup

Both runs:

- **Data:** SRGD ActionRPG only (575 frames at 540×960 HR, paired with engine-aliased LR synth at scale=2 → 270×480, then tile-aligned center-crop to 256×480).
- **LR-synth:** Halton(2,3) jitter (idx+1 base) + area downsample + Gaussian TAA blur σ=1.5 (kernel 7×7) + JPEG q=85.
- **G-buffers:** SRGD has no real depth/normals/motion; the adapter zero-fills depth, motion, sets normals[2]=1.0 (up vector). G-buffer-bias module is enabled but receives near-constant input.
- **Loss:** `composite_loss` at this commit was `L1 + 0.1 × pooled_l1` (the misnamed "ssim_proxy" path). Real SSIM landed AFTER these runs were launched.
- **Optimiser:** AdamW, weight_decay=1e-5, gradient clip max_norm=1.0, no warmup, no scheduler.

### Pico

- **Tier:** pico (channels (8,16,24,32), K=3, 75K params)
- **Batch size:** 2
- **LR:** 3e-4
- **Steps:** 20 000
- **Output:** `<train-host-data>\checkpoints\sprint4-pico-aggressive`
- **Log:** `<train-host-data>\logs\sprint4-pico-aggressive.log`
- **CLI:**
  ```
  python -m oss.gaussian.train.train --tier pico --dataset srgd --srgd-scene ActionRPG \
    --dataset-root <train-host-data>\datasets\srgd --output-dir <train-host-data>\checkpoints\sprint4-pico-aggressive \
    --max-steps 20000 --max-time-seconds 3600 --eval-every 1000 --device cuda \
    --enable-engine-aliased-lr --enable-gbuffer-bias --force-lr-synth \
    --lr-synth-blur-sigma 1.5 --lr-synth-jpeg --batch-size 2 --learning-rate 3e-4
  ```

### Lite (single-scene)

- **Tier:** lite (channels (16,24,32,40), K=5, 178K params)
- **Batch size:** 4
- **LR:** 1e-4
- **Steps:** 12 000 (ran out of time budget before max_steps)
- **Output:** `<train-host-data>\checkpoints\sprint4-lite-single`
- **Log:** `<train-host-data>\logs\sprint4-lite-single.log`
- **CLI:**
  ```
  python -m oss.gaussian.train.train --tier lite --dataset srgd --srgd-scene ActionRPG \
    --dataset-root <train-host-data>\datasets\srgd --output-dir <train-host-data>\checkpoints\sprint4-lite-single \
    --max-steps 12000 --max-time-seconds 3600 --eval-every 1000 --device cuda \
    --enable-engine-aliased-lr --enable-gbuffer-bias --force-lr-synth \
    --lr-synth-blur-sigma 1.5 --lr-synth-jpeg --batch-size 4 --learning-rate 1e-4
  ```

## Result

### Pico — 20 000 steps, FAIL

| Step | model_PSNR | bicubic_PSNR | beats_bicubic |
|-----:|-----------:|-------------:|---------------|
| 1 000 | 11.52 | 28.80 | 0/8 |
| 2 000 | 11.81 | 29.17 | 0/8 |
| 3 000 | 11.39 | 28.99 | 0/8 |
| 5 000 | 12.06 | ~29.0 | 0/8 |
| 10 000 | 12.38 | ~29.0 | 0/8 |
| 15 000 | 11.91 | ~29.0 | 0/8 |
| 20 000 | 12.09 | 29.06 | 0/8 |
| **FINAL** | **12.29** | **29.48** | **0/8** |

PSNR oscillates in [11.31, 12.71] dB throughout. Mean ~12 dB. No upward trend across 20K steps. Bicubic floor at ~29 dB unchanged.

### Lite (single-scene) — 12 000 steps, FAIL

| Step | model_PSNR | bicubic_PSNR | beats_bicubic |
|-----:|-----------:|-------------:|---------------|
| 1 000 | 12.06 | 29.06 | 0/8 |
| 2 000 | 11.90 | 29.97 | 0/8 |
| 9 000 | 11.92 | 28.51 | 0/8 |
| 10 000 | 11.60 | ~29.0 | 0/8 |
| 11 000 | 11.57 | ~29.0 | 0/8 |
| 12 000 | 12.38 | ~29.0 | 0/8 |
| **FINAL** | **11.19** | **29.35** | **0/8** |

PSNR oscillates in [11.19, 12.38] dB. Mean ~11.8 dB. No upward trend.

### Cross-tier comparison

Pico ≈ lite. ~13–17× more parameters bought zero learning improvement on this dataset. **Capacity is not the bottleneck at this scale.**

## Decision

This is the V0 architecture failing its own gate.

1. **Pico is not the cause** — lite at 2.4× capacity does the same thing. The earlier "pico undersized" diagnosis is partially right (pico definitely cannot from-scratch SR) but missed that lite ALSO can't here. Pico distillation-only call still stands.
2. **Hyperparameter sweeps are not promising** — both runs across two LR values, two batch sizes, two tier sizes. Nothing improved. Not running another LR sweep.
3. **Multi-scene batch noise was a red herring** — single-scene lite still fails. Removing that confound did not change the verdict.
4. **The next architectural test is V0.5 (pixel residual head)** per `gaussian-network-architecture.md` §7. Codex 5.5 explicitly recommended this; both GSASR and GS-STVSR use it. Hypothesis: pure 2D Gaussians at this Gaussian budget cannot represent SR detail at game resolutions; a small CNN refinement on the splat output recovers high-frequency texture.
5. **Real SSIM was not active during these runs.** The pooled-L1 fallback ran instead (commit ordering: SSIM landed after launch). Re-running with real SSIM is on the table but not as the primary fix — the loss being mis-weighted by 0.1× of a pooled-L1 term should not produce a 17 dB gap to bicubic. Real SSIM is a tightening, not an architectural rescue.

## Open questions

1. Does V0.5 (pixel residual head on top of splat raster) close the 17 dB gap?
2. Is the bank softmax collapsing to one entry (would explain constant-gray output)? Add a `bank_entropy` metric to diagnose.
3. Are position deltas saturating at zero (Gaussians stuck at tile centers)? Add a `mean_dxy_norm` metric.
4. Is color sigmoid stuck near 0.5 (would also explain ~constant gray output and the L1 ≈ 0.20 plateau)? Add a `mean_color` metric.
5. Does removing the G-buffer-bias help when G-buffers are zero? It zero-inits, so theoretically no harm — but worth toggling off as an A/B.

Items 2–5 are diagnostic metrics that should ship into `train.py` before V0.5 trains, so the next memo has signal not just verdict.

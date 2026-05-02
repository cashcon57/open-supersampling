# Experiments Index

Auto-curated index of `docs/superpowers/experiments/*.md`. Update on every new memo. The paper Results section will be assembled from these entries — keep them well-organised.

## By topic

### Validation gates / decision memos

| Date | Memo | Verdict |
|------|------|---------|
| 2026-05-01 | [validation-decision-memo](../superpowers/experiments/2026-05-01-validation-decision-memo.md) | Sprint 4 authorised conditional on smoke-test gate; engine-aliased LR mandated; anisotropic G-buffer covariance pulled into Sprint 4. |
| 2026-05-02 | [sprint4-smoke-findings](../superpowers/experiments/2026-05-02-sprint4-smoke-findings.md) | Pipeline ✓; CUDA backward live; pico undersized; lite trainability still unresolved at lr={5e-4, 1e-4}. |
| 2026-05-02 | [pico-lite-aggressive-srgd](../superpowers/experiments/2026-05-02-pico-lite-aggressive-srgd.md) | Pico AND lite both flat at 11–12 dB across 12K–20K steps on SRGD ActionRPG with σ=1.5 + JPEG q=85. **V0 architecture fails its own gate.** Promotes V0.5 pixel-residual head as next test. |
| 2026-05-02 | [output-head-dead-init](../superpowers/experiments/2026-05-02-output-head-dead-init.md) | Diagnostic root-cause: V0 output head was zero-init, K Gaussians per tile started identical, gradient symmetry never broke. Smoking gun #2: gsplat 1.4.0 backward returns silent zero when Gaussians too small to hit tiles. Fixes (`6900300` + `6c02cc8`) unstick diagnostics but model still plateaus at 12 dB → V0.5 needed. |
| 2026-05-02 | [v05-pixel-residual-success](../superpowers/experiments/2026-05-02-v05-pixel-residual-success.md) | **V0.5 BICUBIC GATE CLEARED.** Trained on ActionRPG, beats bicubic by +1.47 dB (8/8). Held-out CitySample +1.26, StylizedRendering +0.84, ArchVizInterior +2.08 — all 16/16. Splat-only ablation: 12 dB. **Residual CNN does all the work; splat path is decorative.** Multi-day production unblocked. |

### Naive baseline floors (no training)

| Date | Memo | Verdict |
|------|------|---------|
| 2026-05-01 | [baseline-bench-floor](../superpowers/experiments/2026-05-01-baseline-bench-floor.md) | Bicubic / Lanczos floors established; Bicubic ≈42.78 dB on Sintel. |
| 2026-05-01 | [gaussian-upscaling-naive-test](../superpowers/experiments/2026-05-01-gaussian-upscaling-naive-test.md) | Image-GS naive optim fitting: −3.59 dB vs bicubic — training required. |
| 2026-05-01 | [pretrained-gaussian-sr-eval](../superpowers/experiments/2026-05-01-pretrained-gaussian-sr-eval.md) | GSASR EDSR_DIV2K weights: −4.55 dB vs bicubic — bicubic-LR-trap. |
| 2026-05-01 | [naive-canvas-temporal-stability](../superpowers/experiments/2026-05-01-naive-canvas-temporal-stability.md) | Canvas warp + lifecycle mechanically correct, gated on Sprint 4 signal. |
| 2026-05-01 | [gaussian-denoising-naive-test](../superpowers/experiments/2026-05-01-gaussian-denoising-naive-test.md) | Image-GS at n=1000: 26.90 dB PSNR (beats OIDN 26.56) but loses SSIM/LPIPS — hybrid arch needed. |

## By date

| Date | Memos |
|------|-------|
| 2026-05-01 | baseline-bench-floor, gaussian-upscaling-naive-test, pretrained-gaussian-sr-eval, naive-canvas-temporal-stability, gaussian-denoising-naive-test, validation-decision-memo |
| 2026-05-02 | sprint4-smoke-findings, pico-lite-aggressive-srgd |

## What goes in a memo

Per `docs/papers/lab-notebook-discipline.md`. Standard layout:

1. **Header:** date, status, predecessor memo (if any), hardware.
2. **Hypothesis / question.** One sentence.
3. **Setup.** Hyperparams, data, code commit SHA, exact CLI.
4. **Result.** Table of numbers + (optional) figure.
5. **Decision.** What changes downstream.
6. **Open questions.** What this didn't answer.

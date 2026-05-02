# Experiments Index

Auto-curated index of `docs/superpowers/experiments/*.md`. Update on every new memo. The paper Results section will be assembled from these entries — keep them well-organised.

## By topic

### Validation gates / decision memos

| Date | Memo | Verdict |
|------|------|---------|
| 2026-05-01 | [validation-decision-memo](../superpowers/experiments/2026-05-01-validation-decision-memo.md) | Sprint 4 authorised conditional on smoke-test gate; engine-aliased LR mandated; anisotropic G-buffer covariance pulled into Sprint 4. |
| 2026-05-02 | [sprint4-smoke-findings](../superpowers/experiments/2026-05-02-sprint4-smoke-findings.md) | Pipeline ✓; CUDA backward live; pico undersized; lite trainability still unresolved at lr={5e-4, 1e-4}. |

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
| 2026-05-02 | sprint4-smoke-findings |

## What goes in a memo

Per `docs/papers/lab-notebook-discipline.md`. Standard layout:

1. **Header:** date, status, predecessor memo (if any), hardware.
2. **Hypothesis / question.** One sentence.
3. **Setup.** Hyperparams, data, code commit SHA, exact CLI.
4. **Result.** Table of numbers + (optional) figure.
5. **Decision.** What changes downstream.
6. **Open questions.** What this didn't answer.

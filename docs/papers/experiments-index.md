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
| 2026-05-02 | [splats-cannot-SR-definitive](../superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md) | **Triple-checked: 2D Gaussian splats CANNOT do single-image SR competitively, regardless of implementation.** 5 independent paths converge: (1) direct 50K-Gaussian Image-GS optim −3.59 dB; (2) lite splat-only at multiple lrs −17 dB; (3) lr=1e-1 extreme params −17.47 dB; (4) standard tier −17 dB; (5) `zero+residual` bit-identical to `splat+residual`. Renderer + grad flow audited clean. Pivot recommended: ship V0.5 as a CNN super-resolver OR reorient Gaussian track to denoising (where D1 showed positive). |
| 2026-05-02 | [splats-SR-literature-delta](../superpowers/experiments/2026-05-02-splats-SR-literature-delta.md) | **Literature delta vs. GSASR / GS-STVSR / GaussianSR.** Verdict: we falsified our specific implementation (Gaussians-direct-to-RGB), not the general possibility. Published successes use Gaussians as HR feature extractors (not RGB generators) with mandatory CNN decoders, 20–250× more Gaussians, and attention-based backbones. Our direct-fit test at 50K Gaussians is within the papers' density range yet still loses — density alone is not the unlock. Most likely unlock: redesign Gaussian feat path to be feature-space → CNN decoder (GaussianSR thesis). Recommended first experiment: run released GSASR on our engine-aliased LR to test whether the degradation domain or the architecture gap is the binding constraint. |
| 2026-05-02 | [gsasr-on-engine-aliased-lr](../superpowers/experiments/2026-05-02-gsasr-on-engine-aliased-lr.md) | **GSASR EDSR_DIV2K on 24 SRGD ActionRPG frames with engine-aliased LR (sigma=1.5 + JPEG q=85). Mean PSNR: GSASR 29.136 dB vs bicubic 29.176 dB. Margin: −0.040 dB. GSASR wins 0/24 frames. VERDICT: engine-aliased LR is the binding constraint, not the architecture. Drop-in GSASR does not help. Ship V0.5 CNN; any improvement path requires training on engine-aliased LR, not switching Gaussian SR architecture.** |
| 2026-05-02 | [srcnn-beats-v05-and-gsasr](../superpowers/experiments/2026-05-02-srcnn-beats-v05-and-gsasr.md) | **SR-CNN (clean CNN, no splats) beats V0.5 and GSASR.** Train +2.71 dB; held-out CitySample +2.40, StylizedRendering +1.87, ArchVizInterior +4.05. **64/64 samples beat bicubic.** SR-CNN replaces V0.5 as v0 ship. The day's binding insight: training distribution dominates architecture — any architecture trained on engine-aliased LR beats pretrained models from clean-LR distributions. Splats are dead for SR; the Gaussian track survives in OSS-Gaussian-RR (denoising). |

### Naive baseline floors (no training)

| Date | Memo | Verdict |
|------|------|---------|
| 2026-05-01 | [baseline-bench-floor](../superpowers/experiments/2026-05-01-baseline-bench-floor.md) | Bicubic / Lanczos floors established; Bicubic ≈42.78 dB on Sintel. |
| 2026-05-01 | [gaussian-upscaling-naive-test](../superpowers/experiments/2026-05-01-gaussian-upscaling-naive-test.md) | Image-GS naive optim fitting: −3.59 dB vs bicubic — training required. |
| 2026-05-01 | [pretrained-gaussian-sr-eval](../superpowers/experiments/2026-05-01-pretrained-gaussian-sr-eval.md) | GSASR EDSR_DIV2K weights: −4.55 dB vs bicubic — bicubic-LR-trap. |
| 2026-05-01 | [naive-canvas-temporal-stability](../superpowers/experiments/2026-05-01-naive-canvas-temporal-stability.md) | Canvas warp + lifecycle mechanically correct, gated on Sprint 4 signal. |
| 2026-05-01 | [gaussian-denoising-naive-test](../superpowers/experiments/2026-05-01-gaussian-denoising-naive-test.md) | Image-GS at n=1000: 26.90 dB PSNR (beats OIDN 26.56) but loses SSIM/LPIPS — hybrid arch needed. |

## By date

| Date       | Memos                                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-05-01 | baseline-bench-floor, gaussian-upscaling-naive-test, pretrained-gaussian-sr-eval, naive-canvas-temporal-stability, gaussian-denoising-naive-test, validation-decision-memo  |
| 2026-05-02 | sprint4-smoke-findings, pico-lite-aggressive-srgd, output-head-dead-init, v05-pixel-residual-success, splats-cannot-SR-definitive, splats-SR-literature-delta, gsasr-on-engine-aliased-lr, srcnn-beats-v05-and-gsasr |
| 2026-05-03 | onnx-export-and-bench                                                                                                                                                       |

## What goes in a memo

Per `docs/papers/lab-notebook-discipline.md`. Standard layout:

1. **Header:** date, status, predecessor memo (if any), hardware.
2. **Hypothesis / question.** One sentence.
3. **Setup.** Hyperparams, data, code commit SHA, exact CLI.
4. **Result.** Table of numbers + (optional) figure.
5. **Decision.** What changes downstream.
6. **Open questions.** What this didn't answer.

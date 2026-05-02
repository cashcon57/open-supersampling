# Sprint 4 Smoke-Test Findings — 3080 Ti
**Date:** 2026-05-02
**Status:** Live — multi-tier diagnostic in progress
**Predecessor:** `2026-05-01-validation-decision-memo.md`
**Hardware:** RTX 3080 Ti (12 GB VRAM), Miniconda3 image-gs env, torch 2.4.1 + CUDA 12

This memo captures findings from the first end-to-end Sprint 4 training runs on real game-engine data. Pre-training validation (5 tests, see predecessor) authorised Sprint 4 conditional on a low-capacity smoke gate before any cloud-GPU spend. Cloud spend is now out of budget — MVP must come from this hardware alone — so the smoke test is now load-bearing for the entire training plan.

---

## 1. Pipeline integrity ✅

End-to-end execution works on real data: `SintelGaussianDataset` and `SRGDGaussianDataset` both load, the 12-channel input flows through `GaussianParamNetwork` → `OutputHead` → `Rasterizer` → composite loss, gradients propagate backward, optimiser steps. No NaN or Inf in 5K + 3K + 60+ steps across three runs. Tile-aligned center-crop handles the 540×960 / 270×480 SRGD frames (270 mod 16 ≠ 0) without resampling artefacts.

Bicubic-vs-model evaluation runs at every `--eval-every` interval and at end-of-training; checkpointing succeeds.

## 2. Sintel depth missing on 3080 Ti

`<train-host-data>\datasets\sintel` ships only `clean/` + `flow/`. Sintel's depth supplement is a separate ~1.6 GB download we did not stage. Without depth, `SintelGaussianDataset` fails to form the (frame, depth, flow) triple. Worked around by switching the smoke run to `SRGDGaussianDataset` against `<train-host-data>\datasets\srgd\data\GameEngineData/<scene>` paired with `DownscaleData/<scene>`.

The SRGD adapter now supports both the canonical `<root>/hr/` layout and the on-disk `<root>/data/GameEngineData/<scene>/` layout, with optional `--srgd-scene` filtering.

## 3. CUDA renderer backward IS live (false alarm cleared)

CHANGELOG flagged 2 failing CUDA backward tests in gsplat 1.4.0. Hypothesis was that the smoke-test learning failure (flat PSNR) was caused by silent zero-grad on CUDA. **Falsified.** `scripts/probe_cuda_grad_flow.py` runs a one-step backward on both backends and prints per-leaf gradient L2 norms:

| Leaf | Reference grad_norm | CUDA grad_norm | Ratio |
|------|--------------------:|---------------:|------:|
| `net.stem.conv.weight`        | 5.35e-3 | 1.76e-3 | 3.0× |
| `net.head.weight`             | 4.02e-2 | 7.41e-3 | 5.4× |
| `head.gbuffer_bias.proj.weight` | 2.63e-5 | 3.83e-6 | 6.9× |
| `bank.log_sx`                 | 2.41e-3 | 2.56e-5 | **94×** |

CUDA grads are 3–94× *weaker* than reference grads on this hardware, but every leaf receives non-zero gradient. The gap is consistent with different forward semantics (CUDA gsplat uses tile + topk; reference is full O(N×H×W)) — not with backward bugs. AdamW's per-parameter normalisation absorbs the magnitude difference. **The CUDA backend is usable for training.**

The bank.log_sx 94× reduction is interesting and worth noting — bank parameters get a much weaker training signal on CUDA than on reference, which may explain why the bank was set to `learnable=False` by default. We have not changed that decision.

## 4. Pico tier (75K params) is undersized for SR

5 000-step run on SRGD ActionRPG at lr=3e-4, batch=2, σ=0.5 LR synth (mild):

| Step | Model PSNR | Bicubic PSNR |
|------|-----------:|-------------:|
| 3 000 | 12.20 | 33.64 |
| 4 000 | 11.16 | 34.02 |
| 4 500 | 11.75 | 34.17 |
| 5 000 | 12.56 | 37.52 |

Model output is essentially constant gray (sigmoid(0)≈0.5 with the head zero-init), and 5K AdamW steps barely move it. Pico's 75K parameters cannot represent the SR mapping at 540×960. **Drop pico from the production training table; keep it only for inference on Steam Deck after distillation from a larger trained model.**

## 5. The bicubic-LR-trap is real, even on engine-aliased LR

With the mild default `EngineAliasedLRSynth` (jitter + σ=0.5 TAA blur, no JPEG), the bicubic baseline on SRGD ActionRPG sits at 33–37 dB. That's the upper bound any SR network will achieve, and it's high enough that reaching it requires near-perfect reconstruction — leaving very little gradient for SGD to follow.

Switching to the aggressive synth (σ=1.5 TAA blur, JPEG q=85) drops the bicubic baseline to **26–28 dB**, which finally gives the model a meaningful head-room to optimise into. 2U's "bicubic-LR-trap" warning applies even to "engine-aliased" LR if the synth is too gentle. The trainer's `--smoke-test` mode now hard-overrides the aggressive defaults.

## 6. Lite tier at lr=5e-4 diverges

178 K-param lite tier on multi-scene SRGD (18 031 samples, batch=4, aggressive σ=1.5 + JPEG):

| Step | Model PSNR | Bicubic PSNR |
|------|-----------:|-------------:|
| 1 000 | 13.80 | 28.21 |
| 2 000 | 13.20 | 27.03 |
| 3 000 | **7.97** | 26.14 |

Loss bounces 0.21–0.40 step-to-step with no monotonic decrease. PSNR going *down* across evals is a classic divergence signature. Killed at step 3 280 to retry at lr=1e-4.

Currently in flight: lite tier, lr=1e-4, same data + aggressive synth, 4-hr cap. This should answer "is lite trainable on this dataset at all" before we commit a multi-day standard-tier run.

## 7. Composite loss is misnamed

`composite_loss` in `oss/gaussian/train/train.py` reports a metric called `ssim_proxy`, but the math reduces to `1 − F.l1_loss(avg_pool(rendered, 8), avg_pool(target, 8))`. It is *not* SSIM — it's a pooled L1 difference of luminance, complementary to the per-pixel L1. The total loss is therefore `L1 + 0.1 × pooled-L1`, which is mathematically valid but not what the variable name suggests, and the `0.1×` term is mostly redundant.

We have not changed this yet (training behaviour is unchanged), but it should be renamed or replaced with a real SSIM (e.g. via `pytorch_msssim`) when we revisit the loss.

## 8. Decisions

| Decision | Rationale |
|---|---|
| **Keep CUDA renderer for training** | Probe confirms gradients flow on every leaf. Reference backend OOMs at SR resolution on this card. |
| **Drop pico tier from production training** | 75K params cannot fit 540×960 SR. Lite (178K) or larger only. |
| **Keep aggressive LR synth (σ=1.5 + JPEG q=85) as the smoke default** | Drops bicubic ceiling from ~35 dB to ~26 dB so the training signal is meaningful. |
| **Stay 3080 Ti only** | Lambda H100 spend is out of budget. v0 MVP comes from this hardware. |
| **Hold standard-tier multi-day run until lite-tier stability proven** | Diverging lite at lr=5e-4 means we don't yet know if the architecture trains on this data; spending 24+ hr at standard tier without that signal is reckless. |

## 9. Open questions

1. Does lite at lr=1e-4 stabilise and beat bicubic on at least one scene? *In progress.*
2. Is the loss landscape genuinely too noisy for lr=5e-4, or is the issue elsewhere (gradient clipping threshold, missing warmup, `ssim_proxy` mis-weighting)?
3. Should the bank become `learnable=True` despite the 94× CUDA grad attenuation? Locked-bank training is one less hyperparameter, and AdamW could plausibly compensate.
4. Is multi-scene mixing too high-variance for early training? A single-scene "warm-up phase" before mixing might stabilise the loss landscape.
5. What does standard tier (500 K params) buy us once lite is stable?

These are the questions the lite-at-lr-1e-4 run plus follow-up ablations will answer.

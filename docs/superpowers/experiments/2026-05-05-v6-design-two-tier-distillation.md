# 2026-05-05 — v6 design: two-tier teacher/student distillation

**Status: SUPERSEDED later same day.** This memo proposed a pixel-only architecture (HAT-Base teacher → NAFNet-small + HAT-Tiny students) before the dual-track Gaussian commitment was reaffirmed and before the 2026 Gaussian-temporal research dump (`docs/research/2026-05-05-gaussian-temporal-research-deep-dive.md`) was reckoned with.

**Replaced by:** `2026-05-05-v6-architecture-canonical.md` — Gaussian-temporal foundation with HAT spatial backbone + cross-attention + covariance resampling + S-T variation score pruning + custom kernels per vendor + DLL-shim integration. Three tiers (Pico / Standard / Heavy) — one architecture, scaled.

**What carries over from this memo:** loss recipe (Charbonnier + LPIPS + multi-scale VGG + wavelet L1 + GAN UNetD + edge + temporal consistency), training recipe (AdamW, cosine + warm restarts, EMA, bf16, importance-sampled patches), data plan (TartanAir + Hypersim, no SRGD), 9-channel input (drop SRGD canvas hint).

**What this memo got wrong:** treated Gaussian-temporal as research-only side track instead of architectural foundation; bifurcated by vendor (CNN portable vs transformer NVIDIA-only) instead of using one architecture with custom per-vendor kernels; demoted handheld tier on cross-vendor-runtime grounds that disappear when we hand-write Vulkan compute kernels FSR-style.

Memo retained for forensic value (decision context).

---

(Original content below.)



## Goal

Ship an open super-sampler that:

1. **Cross-vendor portable model** — runs on RX 5500+ / GTX 1660+ / Arc / Apple Silicon at <5 ms for 1080p→1440p. Quality target: between FSR 3 and DLSS 3.
2. **NVIDIA-tier bonus model** — runs on RTX 30+ via TensorRT FP16 at 6-8 ms for 1440p→4K. Quality target: approaches DLSS 4 transformer.

Both students distill from a single teacher. Same training data, same loss recipe, same temporal architecture, different backbone sizes and runtime targets.

## Why this shape

v5 was a single-model bet on RRDB-simple + L1 + LPIPS + temporal head. Honest assessment: that's a 2018-era recipe targeting 2024 quality bars. We can do significantly better by fixing six things simultaneously.

The two-tier shape exists because real-time SR has a fundamental tension: the architectures that produce DLSS-tier quality (transformers) are NVIDIA-tensor-core-dependent for real-time inference, while cross-vendor portability favors CNNs. Trying to satisfy both with one model means compromising both. Two students from one teacher resolves it.

The handheld tier (Steam Deck, integrated GPUs) is deferred — see `2026-05-05-v6-handheld-tier-deferred.md`.

## Architecture

### Tier 0: Teacher

| Component | Choice | Rationale |
|---|---|---|
| Backbone | **HAT-Base** (~17M params) | +0.4-0.6 dB vs SwinIR-S, similar VRAM. Current SOTA classical SR 2023+. |
| Channels | RGB(3) + depth(1) + motion(2) + normals(3) = **9 ch** | Drop the SRGD-era canvas hint. |
| Temporal head | Same as v5 + **t-2 history frame** + **motion-vector refinement CNN** + **learned blend gate** (replace hand-coded depth-test gate) | DLSS / NSRR pattern. v5's gate was the simplest working version, not the best. |

### Tier 1: Desktop-portable student — NAFNet-small

| Component | Choice | Rationale |
|---|---|---|
| Backbone | NAFNet-small (~5M params) | SimpleGate + LayerNorm — fast on all vendors, no attention, ONNX Runtime friendly. |
| Channels | Same 9 | Match teacher input. |
| Temporal head | Same as teacher (light) | Distilled from teacher temporal head. |
| Inference | <5 ms at 1080p→1440p, <10 ms at 1440p→4K | Via ONNX Runtime with vendor-best EP. |
| Runtime path | ONNX Runtime: CUDA EP / DirectML EP / CoreML EP / Vulkan EP | Per-vendor selection at install time. |

### Tier 2: NVIDIA-tier bonus student — HAT-Tiny + TensorRT

| Component | Choice | Rationale |
|---|---|---|
| Backbone | HAT-Tiny (~3M params, smaller windows + fewer blocks) | Closer to teacher's feature space. Distillation gap smaller. |
| Channels | Same 9 | Match teacher. |
| Temporal head | Same | |
| Inference | 6-8 ms at 1440p→4K via TensorRT FP16 (FP8 on Ada+) | Custom fused-attention kernels via FlashAttention-style. |
| Runtime path | ONNX → TensorRT compiled engine, FP16 default, FP8 if Ada+ | RTX-only by design. |

## Loss recipe

| Loss | Weight | Purpose |
|---|---|---|
| Charbonnier | 1.0 | Pixel fidelity. Smoother than L1 near zero (+0.1-0.2 dB). |
| LPIPS (VGG) | 1.0 | Perceptual. |
| Multi-scale VGG L1 | {0.1, 0.1, 1.0, 1.0, 1.0} on relu1_1 / relu2_1 / relu3_1 / relu4_1 / relu5_1 | Perceptual at multiple scales. |
| Wavelet L1 | 0.5 | Explicit high-frequency supervision. Targets the power-line ghost case. |
| GAN (UNetD, hinge) | 0.05, **starts at step 20K** | Sharpness without instability. UNetD = per-pixel real/fake. |
| Edge (Sobel L1) | 0.2 | Sharpness regularizer. |
| Temporal consistency (warp t→t+1, L1) | 0.5 | What v5 already had. Keep. |

GAN warmup: pixel-only first 20K steps, add discriminator after. Stabilizes training significantly.

## Training recipe

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.99), wd=1e-4 |
| LR | 2e-4 cosine + 3 warm restarts, T_0=50K, T_mult=1 |
| Precision | bf16 (Ampere+ supports it natively, more stable than fp16 for GAN) |
| Effective batch | 16 (batch=4, accum=4) |
| Patch size | 256² |
| Patch sampling | 70% importance-sampled (variance-weighted), 30% uniform |
| EMA | β=0.999 |
| Steps (teacher) | 300K |
| Steps (student) | 80K each, with KD loss + GT loss |

KD loss for students: feature-matching at intermediate teacher layers + output L1 to teacher prediction + ground-truth supervision. Temperature T=4 on soft targets.

## Data

| Dataset | Use | Held-out env |
|---|---|---|
| TartanAir | 60% of training mix | `oldtown` |
| Hypersim | 30% of training mix | random 10% of scenes |
| (eval) | 10% — held-out from both | combined eval set |

**No SRGD.** SRGD's zero G-buffers create distribution-mix problems. v6 commits to "real G-buffers everywhere or nothing."

## VRAM budget on 3080 Ti 12 GB

| Run | Crops | Batch | GAN | EMA | Estimated peak |
|---|---|---|---|---|---|
| Teacher (HAT-Base) | 256² | 4 | yes (UNetD) | yes | ~10.5 GB ✓ |
| Student-portable (NAFNet-small + frozen teacher) | 256² | 4 | yes (light) | no | ~9 GB ✓ |
| Student-NVIDIA (HAT-Tiny + frozen teacher) | 256² | 4 | yes (light) | no | ~10 GB ✓ |

3080 Ti carries the whole pipeline. No GPU rental needed for development.

A100 80GB rental ($100, ~30 h) considered for the final teacher run before showing the model publicly — decided based on whether 3080 Ti numbers look promising at 100K steps.

## Wall-time estimate

| Phase | Wall time on 3080 Ti |
|---|---|
| HAT-Base impl + smoke test | 1-2 days |
| NAFNet-small impl (likely partial port from existing repos) | 1 day |
| HAT-Tiny variant (parameter knock-down of HAT-Base) | 0.5 day |
| Teacher training (300K steps) | ~50-60 h |
| Student-portable distillation (80K steps) | ~15 h |
| Student-NVIDIA distillation (80K steps) | ~15 h |
| TensorRT export + verify | 1-2 days |
| **Total** | **~2 weeks** |

## Ship order

1. **v6-teacher** — first deliverable (used internally to validate quality ceiling, never user-facing)
2. **v6-portable** — first user-facing release (most users, broadest hardware support)
3. **v6-NVIDIA-bonus** — follow-up, "for enthusiasts" tier

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| HAT-Base impl bugs | high | Smoke-test at 1K steps before committing to 300K |
| GAN instability | medium | bf16, hinge loss, UNetD, GAN warmup at 20K. Monitor D-loss. |
| Distillation gap larger than expected | medium | Have NAFNet-small student converge to ~85% teacher; HAT-Tiny student to ~92%. If lower, increase student capacity. |
| TensorRT export breaks on HAT attention | medium-high | Reference impls exist (FlashAttention has TensorRT plugin). Fall back to ORT-CUDA EP if blocked. |
| 2-week estimate slips to 4 | medium | Honestly likely. Plan for 3-4 weeks elapsed. |

## Out of scope for v6

- Handheld tier (Steam Deck, integrated GPUs) — see deferred-work doc
- Distillation across content domains (e.g., separate models for photoreal vs stylized)
- More than 9 input channels (albedo, roughness, metallic) — wait for INSANE-mode capture data
- Real-time engine integration / DLL hooks beyond what v5 already has
- Multi-frame teacher (>2 prev frames) — keep at t-1 + t-2 for v6, expand in v7

## Followups for v7

- Recurrent latent state (carry features forward across frames, not just RGB outputs)
- More than 9 channels once INSANE-mode capture data accumulates
- Handheld tier revisit (see deferred doc)
- Domain-specific student variants if quality varies meaningfully across content types

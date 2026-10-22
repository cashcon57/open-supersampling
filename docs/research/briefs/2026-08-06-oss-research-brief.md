# OSS Research Brief — 2026-08-06

_Collection: local Git/README, rendered public dashboard, GitHub issues/repository search, arXiv Atom plus abstract-page verification, Archive.org Advanced Search, primary vendor pages, and Google News RSS. RSS-only signals are labelled as leads._

## Project Status Summary

- **Latest commits:** `a97a180`/`fcf8084` add tailnet-gated training controls and control API/dashboard UI. The preceding series preserves the v7 step-5K recovery point; fixes WSL2 DataLoader/cross-OS resume; and enables FlashAttention 2, TF32 convolution, and prefetching.
- **Open issues (12):** top operational blockers remain #14 VRAM measurement, #13 Nsight L2 profiling, #12 packed low-rank canvas, #3 tile bin, and #4/#9 student distillation/reparameterization. The full list still includes cross-vendor kernels (#8), FSR 4 teacher signal (#7), and the no-ML variant (#5).
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is the HAT-Tiny teacher with native `(x,y,t)` Gaussians, `V_xt` covariance motion, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The ≤1M TensorRT-FP8 student and all non-CUDA paths remain unbuilt. Published held-out quality remains v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered):** headline says “training right now,” but metrics remain **step 36,650 / 100,000**, loss **0.19268**, viz **36,000**. Trainer health now says **offline, stuck for 34h**; watcher is publishing, CF Worker is HTTP 200/97 ms, R2 is healthy, and GPU stats are unavailable. This confirms no training progress; public delivery is healthy while trainer/metric emission is not. The simultaneous “live” copy remains contradictory/stale.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields](https://arxiv.org/abs/2608.04581)** (arXiv:2608.04581; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko; 2026-08-05) — Dynamically varies Gaussian cardinality and feature-channel budgets per anchor according to local spatial-temporal complexity; reports up to 1.5× higher compression than prior anchor-based approaches at comparable quality on difficult MPEG sequences. **Why it matters to OSS:** **HIGH RELEVANCE** direct ablation/reference for v7 parent-child density and #12 packed-canvas budgeting—allocate both primitive count and feature capacity from motion/detail confidence rather than using a uniform per-parent child budget.
- **[FAST-GS](https://arxiv.org/abs/2608.01958)** (arXiv:2608.01958; Zhengyang Zhang et al.; 2026-08-03; previously tracked) — Fourier motion components plus frequency-aware regularization target high-frequency motion and long-term dynamic-GS stability. **Why it matters to OSS:** use only as a teacher-side hard-motion regularizer ablation; exact engine MVs remain the stronger motion input.

### Super-Resolution / Upscaling

- **[CLEAR](https://arxiv.org/abs/2608.02206)** (arXiv:2608.02206; Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03; previously tracked) — Routes high-frequency gradients/densification using evidence and patch reliability rather than uniformly trusting weak LR observations. **Why it matters to OSS:** direct design for confidence-gating parent-child densification and teacher losses where LR color, depth, and MV evidence disagree.
- No newly material real-time game-SR/student paper cleared the bibliography gate. Existing NanoVSR remains the practical fully-convolutional reparameterized student control.

### Frame Extrapolation / Generation

- No credible new single-network, exact-motion-vector, <2 ms extrapolation paper appeared. Broad hits were diffusion/world-model work rather than an OSS-FX shipping competitor. The key competitive risk remains internal: OSS needs held-out future-frame metrics before making the unified-SR/FG claim.

### Inference Optimization

- **[DeGS](https://arxiv.org/abs/2608.02099)** (arXiv:2608.02099; Minnan Pei et al.; 2026-08-03; previously tracked) — Separates alpha/transmittance parsing from blending and reorganizes irregular splat work into denser stages. **Why it matters to OSS:** strongest current hypothesis for #3/#12/#13, but validate CUDA/HIP transfer with raster latency, L2, DRAM bytes, occupancy, and parity—not its accelerator speedup headline.
- No new material TensorRT/INT8-SR/kernel-fusion paper was found beyond already-tracked work.

## Archive.org Findings

- Advanced Search found only three archived GitHub/star-list snapshots for `"gaussian splatting" super-resolution`; no validated paper copy. `"frame generation" deep learning` was FluidFrames/DLSS-Swapper builds plus unrelated diffusion; `"real-time upscaling" game` returned zero.
- Known-work queries remain noisy: GS-STVSR/STSS returned zero; GSASR returned an unrelated CIA item; GRAPE was overwhelmingly unrelated. No archived version of GS-STVSR, GSASR, GRAPE, or STSS was verified. This is not evidence of absence.

## GitHub / Open Source

- No new direct OSS competitor or open DLSS/FSR analysis tool surfaced. Search hits were mostly off-domain or zero-star noise.
- **[RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** is a meaningful active baseline, not a release-level threat: GPL-3.0, 15 stars, native DX12/HLSL runtime, G-buffer guidance, exported FP16/INT8 weights, and a July 23 RX 7900 XTX WMMA INT8 1×1-convolution validation. Its Linux/NVIDIA runtime is planned and shaders remain WIP. **Action:** inspect data contract/test methodology only; preserve GPL isolation.
- **[BananaScaler](https://github.com/julesklord/bananascaler)** is a new MIT GPU video-upscaler TUI, but has no game/MV/canvas relevance. No action.

## News

- **NVIDIA / DLSS:** no new primary announcement in the last 14 days. NVIDIA’s May DLSS 4.5 UE plugin announcement remains the quality/integration bar: Dynamic Multi Frame Generation, 6× MFG, and second-generation transformer SR—still a two-path vendor stack, unlike OSS’s intended unified canvas.
- **AMD / FSR:** no new primary announcement in the window. AMD’s current FSR SDK 2.3 page exposes FSR Upscaling 4.1.1 on RDNA 3 discrete GPUs, while its UE plugin remains on 4.0.3 and UE 5.2–5.7. That compatibility lag is an integration opportunity, not a reason to change OSS architecture.
- **Intel / Apple / engines:** an RSS-only report says XeSS plugin 3.1 adds UE 5.8 support; no Intel primary confirmation found. UE 5.8 is available but no new native upscaling feature was identified. No material recent Apple/Metal or Unity upscaling announcement was verified.

## Action Items / Follow-ups

- [ ] **Critical:** restore the 3080 Ti trainer or metric emitter. Require a higher step, fresh GPU telemetry, and a new visualization before dashboard copy calls the run live.
- [ ] **Deep-read ACA-GS:** compare adaptive anchor cardinality + feature masking against v7’s parent-child density; define a fixed packed-memory budget and evaluate temporal PSNR/LPIPS and splat count.
- [ ] Prototype CLEAR-style confidence gates for densification/teacher gradients using LR color, depth, and engine-MV reliability.
- [ ] After #13, profile a DeGS-like parse/reorganize/blend split around the tile bin; reject it unless it improves measured latency and bandwidth without image drift.
- [ ] Verify XeSS UE 5.8 support through Intel primary release notes; keep it as SDK-compatibility monitoring only.

## Sources

- arXiv: [ACA-GS](https://arxiv.org/abs/2608.04581), [FAST-GS](https://arxiv.org/abs/2608.01958), [CLEAR](https://arxiv.org/abs/2608.02206), [DeGS](https://arxiv.org/abs/2608.02099)
- Vendor/engine: [NVIDIA DLSS 4.5 UE update](https://developer.nvidia.com/blog/whats-new-for-game-developers-in-nvidia-rtx-dlss-4-5-for-ue5-and-multilingual-ai-characters/), [AMD FSR Upscaling](https://gpuopen.com/amd-fsr-upscaling/), [UE 5.8](https://www.unrealengine.com/news/unreal-engine-5-8-is-now-available)
- Open source: [RDNU](https://github.com/Adam-AtlasSoftware/RDNU)

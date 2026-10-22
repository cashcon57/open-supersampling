# OSS Research Brief — 2026-07-30

_Collection: local Git/README, GitHub REST, rendered training dashboard, arXiv category/abstract pages (Atom API rate-limited), Archive.org Advanced Search, GitHub repository search, primary NVIDIA/AMD pages, and Google News RSS. RSS-only items are labelled as leads._

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard/API surface. The preceding commits preserve the v7 Pico step-5K recovery snapshot, fix cross-OS resume and WSL2 loader deadlock, and add FlashAttention/TF32/bf16 rasterizer safeguards.
- **Open issues:** **12**. Immediate gates remain #14 (v6.2 VRAM measurement), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #9 (reparameterized student), #8 (cross-vendor kernels), #7 (FSR 4 teacher signal), and #4 (≤1M student distillation).
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is the active HAT-Tiny *teacher*: N-D `(x,y,t)` Gaussians, `V_xt` motion covariance, parent-child density, Mip-Splat filtering, and alpha=0.5 OSS-FX supervision. The shipping ≤1M TensorRT FP8 student and every non-CUDA backend remain unimplemented/untrained. Published held-out metrics remain v5-only: **25.703 dB PSNR / 0.1666 LPIPS** on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-30):** headline reports `36,650 / 100,000`, loss `0.19268`, visualization `36,000`, and ETA ~1.1d. Service health says **Trainer offline, stuck at step 36,650 for 36h**; watcher is healthy (published 0s ago), CF Worker is HTTP 200 in 94ms, and R2 is healthy. GPU telemetry is unavailable. Its live/ETA copy conflicts with health and unchanged step: this is a trainer/metrics outage, not training progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[AtlasLC: Fast Codec-Ready Compression of Object-Centric 3D Gaussian Splatting](https://arxiv.org/abs/2607.26525)** (arXiv:2607.26525v1; ByungHyun Kim, Jinwoo Jeon, Woontack Woo; 2026-07-29) — Training-free local-competition pruning plus deterministic atlas packing targets payload, decode latency, FPS, and geometric fidelity rather than image-space rate-distortion alone. **Why it matters to OSS:** direct input to #12’s packed canvas layout: test its sort/packing and deployment-aware scorecard against the current low-rank layout, while retaining OSS’s screen-space/time-axis constraints.
- **[SpatialQ: Understanding 3D Gaussian Splatting Scene Quality via Visual-based MLLM](https://arxiv.org/abs/2607.26595)** (arXiv:2607.26595v1; Jingxuan Su, Shenglin Wang, Tiesong Zhao, Ge Li, Wei Gao; 2026-07-29) — learns a multi-view, depth/point-cloud-aware 3DGS quality representation. **Why it matters to OSS:** off-domain and not a runtime design, but a useful lead for a multi-signal visual-regression gate; do not add an MLLM to the runtime or training loop.
- **[SplatStream: Fine Granular Scalable Gaussian Splatting for Adaptive 3D Scene Streaming](https://arxiv.org/abs/2607.25971)** (arXiv:2607.25971v2; Muhammad Talha, William Gordon, Sajid Umair, Zhu Li, Anique Akhtar, Joel Jung; submitted 2026-07-28, revised 2026-07-29) — dynamic 3DGS with importance-layered packets, temporal B-frames, and inter-layer prediction. **Why it matters to OSS:** directly informs #12 importance ordering, temporal deltas, and quality-layer budgeting; already tracked.
- The original targeted Atom queries returned 429/timeouts. Direct category-page fallback found the two July-29 papers above; no newly verified direct 2D-GS/arbitrary-scale SR paper was found after the July-29 brief.

### Super-Resolution / Upscaling

- **[TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution](https://arxiv.org/abs/2607.22231)** (arXiv:2607.22231v1; Sicheng Gao, Zhuyun Zhou, Yixuan Liu, Tong Shen, Zongwei Wu, Radu Timofte; 2026-07-24) — routes motion/structure-important regions through a higher-fidelity stream. **Why it matters to OSS:** teacher-side reference for motion/boundary-driven Gaussian allocation, never the ≤2ms shipping path; already tracked.
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (arXiv:2607.10495v1; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11) — structurally reparameterized VSR that collapses to ordinary convolutions. **Why it matters to OSS:** closest direct control for #9/#4 student backbone, deployment graph, and quantization experiments; already tracked.

### Frame Extrapolation / Generation

- No newly verified material single-network game-frame-extrapolation paper. Search results were mainly diffusion, codec, or event-camera work; retain GFFE/ANVIL as OSS-FX baselines and do not treat codec motion vectors as equivalent to exact engine MVs.

### Inference Optimization

- AtlasLC is the only new material deployment/inference lead. Its reported 25× atlas-preparation and 5× end-to-end compression improvements motivate measurement, not adoption: profile bytes/Gaussian, sort/pack time, L2 hit rate, DRAM traffic, and visual error on OSS’s dynamic screen-space canvas.
- No new material TensorRT/INT8/kernel-fusion SR paper cleared the relevance gate. #13 profiling remains the prerequisite for #12 layout claims, followed by QAT/reparameterization measurements against NanoVSR-like student graphs.

## Archive.org Findings

- Advanced Search returned **3** noisy GitHub-list snapshots for `"gaussian splatting" super-resolution`; **3** mixed items for `"frame generation" deep learning` (DLSS Swapper/FluidFrames artifacts plus an unrelated video-generation registration); and **0** for `"real-time upscaling" game`.
- No confirmed archived/taken-down update for GS-STVSR, GSASR, GRAPE, or STSS surfaced. Prior targeted Wayback CDX requests returned HTTP 400, so this is an incomplete archive result—not evidence of absence.

## GitHub / Open Source

- Fresh GitHub REST searches found no credible new Apache-compatible DLSS/FSR/XeSS analysis tool or OSS-runtime competitor. `hvcl/3DCT_SR_DDNM_NAB-GS` is Apache-2.0 but CT-specific; not a runtime lead.
- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** remains the active compatibility/analysis surface across DLSS/XeSS/FSR/FG (GPL-3.0). Use it only for black-box input-contract and presentation-timing tests; no code reuse.
- **[RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** remains a small GPL-3.0 RDNA3 G-buffer-guided upscaler lead (15 stars, pushed July 24). Inspect its data contract/evaluation setup only.

## News

- **NVIDIA — verified primary:** NVIDIA’s July 28 post says Halo: Campaign Evolved ships with DLSS Super Resolution + Frame Generation; Palworld 1.0 and Killing Floor 3 can be upgraded to DLSS 4.5 via the NVIDIA app, while Corsair Cove launches with DLSS Multi Frame Generation. **Threat:** vendor SR/FG deployment continues to broaden rapidly; OSS needs a reproducible exact-MV unified-SR+FX benchmark rather than feature-count comparisons.
- **AMD — verified primary:** GPUOpen’s current SDK 2.3 announcement says FSR Upscaling 4.1.1 is available on RX 7000-series hardware, with Frame Generation 4.0.1 and Ray Regeneration 1.2 improvements for RDNA 4. Its WMMA guides also document the importance of fused low-precision matrix paths on RDNA 4. **Opportunity:** make #8’s HIP/WMMA portability plan explicitly measure fused FP8/INT8 layouts, not just CUDA parity.
- **Intel / Apple / Unreal / Unity:** no fresh primary technical upscaler material was verified. Google News contained DLSS 5/DLSS 4.5 claims, but except for NVIDIA’s own DLSS 4.5 integration post they remain RSS-level leads, not architecture inputs.

## Action Items / Follow-ups

- [ ] **Critical:** repair/restart the 3080 Ti trainer or its metrics emitter. Confirm a step increase, fresh GPU telemetry, and a new visualization before marking recovery; watcher/CDN health is insufficient.
- [ ] Deep-read AtlasLC, then add a #12 microbenchmark: deterministic/sort-based packing plus local competitive pruning versus packed low-rank canvas. Measure bytes/Gaussian, packing time, L2 hit rate, DRAM traffic, raster time, and PSNR/temporal error.
- [ ] Keep #13 ahead of layout claims; use AMD’s current WMMA guidance to define HIP FP8/INT8 fused-kernel acceptance criteria for #8.
- [ ] Use TRaM-VSR only to derive teacher-side motion/boundary allocation targets; ablate against v7 loss-adaptive density rather than adding diffusion runtime.
- [ ] Build a compatibility-only OptiScaler matrix around FSR 4.1.1, FSR/XeSS/DLSS input contracts, FG presentation timing, and GPL isolation.
- [ ] Define the OSS versus FSR 4.1/FG 4 benchmark: same scene, LR/HR, MVs/G-buffers, latency, PSNR/LPIPS/temporal error, and extrapolation quality.

## Bibliography Update

- Added AtlasLC. SpatialQ is a QA/evaluation lead but does not clear the architecture/training/inference relevance threshold for the bibliography.

# OSS Research Brief — 2026-08-27

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` is the only commit in the local ten-commit window.
- **Open issues:** **12** open. Immediate execution gates remain packed low-rank canvas layout / VRAM / L2 profiling (#12–#14), then tile binning (#3), student distillation (#4/#7/#9), and cross-vendor kernels (#8).
- **README snapshot:** OSS remains pre-alpha. v7 is the N-D `(x,y,t)` Gaussian-canvas **teacher**; the ≤1M shipping student and native CUDA/HIP/Metal/Level Zero/Vulkan kernels are not built.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **step 36,650 / 100,000**: total loss **0.19268**, SR Charbonnier **0.02360**, FG Charbonnier **0.07643**, and **2,304** canvas elements. The live `status.json` check at `2026-08-28T00:04:48Z` reports Trainer **offline**, “stuck at step 36650 for 97h”; Watcher, CF Worker, and R2 are healthy. That is no training progress since yesterday; the dashboard’s “training now” history copy is stale/conflicting.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- The arXiv Atom API returned HTTP 429 for all targeted Gaussian-temporal, VSR/VFI, and inference searches. No newly submitted paper was promoted without primary metadata.
- **CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution** (arXiv:2608.02206; Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03) — Evidence-conditioned routing anchors on observed LR gradients and only routes high-frequency/densification updates where warranted. **Why it matters:** already tracked; a strong control for v7 disocclusion spawning and imperfect-teacher confidence gating, subject to temporal-identity tests.
- **QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting** (arXiv:2608.18285; Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, Kevin C. Zhou; 2026-08-18) — Quantization-aware anchor deformation and change-gated densification target compact dynamic-GS residuals. **Why it matters:** already tracked; useful #12 packed-canvas/residual-layout control, not a direct game-SR raster design.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11) — A structurally reparameterized VSR CNN that collapses to standard convolutions for deployment. **Why it matters:** already tracked and the clearest external student-backbone candidate for #4/#9; benchmark a fused ≤1M graph at OSS 1080p-LR, not against Jetson/REDS numbers.
- **TRaM-VSR** (arXiv:2607.22231; Sicheng Gao et al.; 2026-07-24) — Motion-sensitive token routing/merging for one-step diffusion VSR. **Why it matters:** importance allocation is a teacher-side/offline training lead only; diffusion/token routing misses the ≤2 ms student target.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (arXiv:2608.10423; Sebastian Künzel et al.; 2026-08-11) — A sparse layered image-space cache predictively refreshes shading/disocclusions, reporting 60 Hz shading to 240 Hz display extrapolation. **Why it matters:** already tracked; strongest systems control for OSS-FX visibility/disocclusion and tile-refresh behavior. It threatens the “near-zero marginal extrapolation” claim if OSS lacks explicit stale/visibility handling.
- **MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation** (arXiv:2606.27718; Jun-Sang Yoo, Seung-Won Jung; 2026-06-26) — Flow-aligned scan trajectories allocate refinement by motion magnitude. **Why it matters:** a residual-refinement/confidence-gating ablation after exact engine-MV warp, not a replacement for fractional-time Gaussian rendering.

### Inference Optimization
- No new TensorRT/INT8/fused-SR result cleared the materiality gate; arXiv primary search was rate-limited. Keep #12 packed layout and #13 Nsight L2 measurements ahead of decisions on attributes, precision, sort keys, or fusion.
- **3D Gaussian Accelerated Ray Tracing** (arXiv:2608.17298) and **Differentiable Voronoi Ray Tracing Beyond Rasterization Speeds** (arXiv:2608.17682) remain already-tracked renderer controls. Neither changes the rasterized shipping decision without OSS-specific quality/latency data.

## Archive.org Findings
- Advanced Search returned only three noisy GitHub snapshot artifacts for `"gaussian splatting" super-resolution`, three miscellaneous artifacts for `"frame generation" deep learning`, and zero results for `"real-time upscaling" game`.
- Targeted `GS-STVSR` returned no record; `GSASR` returned an unrelated CIA document; `GRAPE` is heavily acronym-noisy. No verified Archive.org copy of GS-STVSR, GSASR, GRAPE, or STSS was found. This is limited archive coverage, **not** evidence of withdrawal or removal.

## GitHub / Open Source
- **IE-SRGS** was pushed Aug 27 and describes 3D Gaussian-SR, but has **1 star** and no asserted SPDX license. Watchlist only pending reproducibility/runtime inspection.
- **OSSS** (MIT; pushed Aug 25) is a zero-dependency Windows window-level upscaler/frame generator. **ZeroFG** (Apache-2.0; pushed Aug 13) is an experimental Vulkan mobile FG engine. Both have negligible adoption and no demonstrated DLSS/FSR-class quality or latency.
- No credible new open-source DLSS/FSR analysis tool surfaced in REST searches.

## News
- **NVIDIA released DLSS 4.5 Ray Reconstruction on Aug 25.** The new second-generation transformer RR model is unified with DLSS Super Resolution and analyzes temporal/spatial engine data; NVIDIA says it has 20% more parameters with similar performance, adds more spatial awareness, and exposes global/per-game NVIDIA App model controls. This is the material competitive development in the last 14 days.
- **OSS implication:** maintain model versioning/capability manifests and per-title temporal-accumulation/quality controls. NVIDIA’s exposed model-selection UX makes a single opaque OSS setting less competitive, even before comparing image quality.
- NVIDIA also states RTX Remix now supports DLSS 4.5 RR and Multi Frame Generation, making the modding ecosystem a practical test surface for OSS integration expectations.
- No dated primary AMD FSR 4, Intel Arc/XeSS, Apple MetalFX, Unreal, or Unity upscaling announcement was verified in this window. RSS-only third-party leads were not promoted to product facts.

## Action Items / Follow-ups
- [ ] **P0 operations:** investigate/restart or explicitly close the trainer; the same step now carries a 97h stuck state while publication services remain healthy.
- [ ] **#12/#13 gate:** measure packed residual/anchor layouts and L2 behavior before committing canvas attribute formats, quantization, or kernel fusion.
- [ ] **OSS-FX control:** use Amulet as the explicit disocclusion/stale-shading baseline; add visibility age, tile refresh, and frame-pacing telemetry alongside PSNR/LPIPS.
- [ ] **Student experiment:** port NanoVSR-style reparameterizable standard-convolution blocks as one #9 candidate; profile the fully fused ≤1M graph at 1080p-LR.
- [ ] **Product positioning:** prototype model-version/capability manifests and per-title temporal controls to meet the new DLSS 4.5 integration/UX bar.
- [ ] **Competitor watch:** inspect IE-SRGS licensing/runtime; monitor OSSS and ZeroFG only for measured frame pacing, latency, and real D3D/Vulkan input integration.

## Sources
- https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
- https://github.com/cashcon57/open-supersampling/blob/main/README.md
- https://opensupersampling.com/data.json
- https://opensupersampling.com/status.json
- https://arxiv.org/abs/2608.02206
- https://arxiv.org/abs/2608.18285
- https://arxiv.org/abs/2607.10495
- https://arxiv.org/abs/2608.10423
- https://archive.org/advancedsearch.php
- https://github.com/ZheXues/IE-SRGS
- https://github.com/Lenovive/OSSS
- https://github.com/hy300leosquizz-ctrl/ZeroFG
- https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements/

# OSS Research Brief — 2026-08-26

## Project Status Summary
- **Latest commits:** local history contains only `b44278a test: add local Chrome Playwright dashboard smoke`; no newer model, training, or kernel commit is present.
- **Open issues:** **12** open. Immediate execution gates remain packed low-rank canvas layout/VRAM/L2 measurement (#12–#14), then tile binning (#3), student/distillation (#4/#7/#9), and cross-vendor ports (#8).[1]
- **README snapshot:** OSS is still pre-alpha. v7 is the N-D `(x,y,t)` Gaussian-canvas *teacher* architecture; the ≤1M shipping student and native CUDA/HIP/Metal/Level Zero/Vulkan kernels are not built.[2]
- **Training dashboard:** `srcnn-v7.0-pico-005` is unchanged at **step 36,650**, 2,304 canvas elements, total loss **0.19268**, SR Charbonnier **0.02360**, and FG Charbonnier **0.07643**.[3] The status endpoint reports Trainer **offline**, “stuck at step 36650 for 73h,” while Watcher, Worker, and R2 are healthy.[4] Its `checked_at` is `2026-08-27T00:01:24Z` although this brief is dated Aug 26; record it as endpoint-provided metadata. The dashboard history still says “training now,” so that copy is stale/conflicting—not evidence of progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- No newly submitted post-Aug-25 paper was verified. The arXiv Atom category/keyword collector timed out twice with zero response bytes; direct abstract pages were used for material candidates instead.

### Super-Resolution / Upscaling
- No new real-time game-SR or ≤1M student/distillation result cleared the relevance gate. Prioritize #12 packed-layout and #13 Nsight data before choosing final attribute precision, key width, or fusion boundaries.

### Frame Extrapolation / Generation
- No new verified frame-extrapolation paper changes the current plan. **Amulet** remains the strongest already-tracked systems control for OSS-FX: persistent layers plus predictive shading and tile refresh explicitly test the visibility/disocclusion gap in pure motion warp.

### Inference Optimization
- **3D Gaussian Accelerated Ray Tracing: Fast training through particle-based backward propagation** — Laurent Vit, Oliver Batchelor, Richard Green; 2026-08-18; **arXiv:2608.17298**. It maps a primitive-centric particle-style backward pass to tiles to reduce splat-gradient synchronization pressure.[5] **Why it matters to OSS:** a training-kernel control for v7’s temporal Gaussian primitives and canvas-gradient accumulation, not a replacement for the rasterized shipping path. This is newly material (not newly published) and is added once to the bibliography.
- **Differentiable Voronoi Ray Tracing Beyond Rasterization Speeds** — Bernardo Taveira, Carl Lindström, Joakim Johnander, Fredrik Kahl; 2026-08-18; **arXiv:2608.17682**. Its fixed-budget, locality-oriented differentiable renderer is a useful renderer/kernels comparator, including motion-blur support.[6] **Why it matters:** a bounded-work alternative to compare against the Gaussian rasterizer; do not import it into the v7 path without measured image-quality and latency evidence.
- **GaussVid: Sparse-View Gaussian Splatting with 3D-Aware Video Diffusion Priors** — Xinhui Liu, Can Wang, Wei Jiang, Wei Wang, Dong Xu; 2026-08-22; **arXiv:2608.21849**. It uses a camera-conditioned video-diffusion prior for sparse-view Gaussian reconstruction.[7] **Why it matters:** offline teacher/data-generation lead for temporal consistency or difficult-hole augmentation only; diffusion is not a viable ≤2 ms inference mechanism.

## Archive.org Findings
- Broad Archive Advanced Search for `"gaussian splatting" super-resolution`, `"frame generation" "deep learning"`, and `"real-time upscaling" game` produced only noisy catalog/software artifacts, not confirmed paper copies.[8]
- Targeted `GS-STVSR` returned no record. `GSASR` returned an unrelated 1961 CIA document; `GRAPE` and `STSS` are acronym-noise queries. No verified Archive.org copy of GS-STVSR, GSASR, GRAPE, or STSS was found; this is limited archive coverage, **not** evidence that a paper was withdrawn. Targeted arXiv Wayback requests timed out and remain unverified.

## GitHub / Open Source
- No new direct OSS competitor appeared in GitHub REST searches using `pushed:>=2026-08-25`. `OSSS` is still the already-tracked MIT Windows/Direct3D11 optical-flow FG/upscaler project, pushed Aug 25—not a quality or latency delta.[9]
- **splatx-metal** is a new native Apple-Silicon 3DGS training/rasterization codebase (Apache-2.0; pushed Aug 26). It is medium-relevance cross-vendor Gaussian-canvas reconnaissance, not a temporal SR/game-FG competitor.[10]
- No new open-source DLSS/FSR analysis tool appeared. `IE-SRGS` remains a watchlist lead pending license, provenance, and measured runtime inspection.

## News
- **NVIDIA released DLSS 4.5 Ray Reconstruction** via a new NVIDIA App update. NVIDIA describes a second-generation transformer RR model, unified with DLSS Super Resolution and temporal/spatial engine data; it is available across GeForce RTX GPUs.[11] This is the material 14-day vendor development.
- **OSS implication / competitive signal:** NVIDIA’s model-versioned, app-delivered reconstruction and per-game/global model controls raise the integration bar. OSS should treat models as versioned/capability-gated artifacts and preserve per-title override plus temporal-accumulation controls—not one opaque “upscaler” setting.
- No dated primary AMD FSR 4, Intel Arc/XeSS, Apple MetalFX, Unreal, or Unity upscaling announcement was verified in this window. Third-party news leads were not elevated to product facts.

## Action Items / Follow-ups
- [ ] **P0 ops:** investigate/restart or explicitly close the trainer state. Healthy publication services are not training progress.
- [ ] **Training-kernel control:** evaluate particle/tile-mapped backward accumulation from arXiv:2608.17298 against current v7 gradients; measure atomic contention, memory, and step time before architecture changes.
- [ ] **Implementation gate:** resolve #12 then #13 before committing to packed attributes, sort-key width, quantization, or fused-kernel boundaries.
- [ ] **OSS-FX product control:** keep Amulet’s explicit visibility/refresh behavior as the disocclusion baseline; add present-interval/frame-pacing telemetry alongside image metrics.
- [ ] **Platform positioning:** prototype model-version/capability manifest plus per-title temporal controls; this is now a concrete competitive UX requirement, not merely SDK polish.
- [ ] **Competitor watch:** inspect IE-SRGS license/measurements and splatx-metal’s Metal execution model; monitor OSSS for actual latency and frame-pacing evidence, not README claims.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/data.json
[4] https://opensupersampling.com/status.json
[5] https://arxiv.org/abs/2608.17298
[6] https://arxiv.org/abs/2608.17682
[7] https://arxiv.org/abs/2608.21849
[8] https://archive.org/advancedsearch.php
[9] https://github.com/Lenovive/OSSS
[10] https://github.com/PoSTMEDIA-AI/splatx-metal
[11] https://www.nvidia.com/en-us/geforce/news/gamescom-2026-nvidia-geforce-rtx-dlss-4-5-announcements/

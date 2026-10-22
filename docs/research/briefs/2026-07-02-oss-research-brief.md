# OSS Research Brief — 2026-07-02

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated training controls; `fcf8084` training control API/dashboard panel; `0660f55` preserved v7-pico-005 step-5000 snapshot/recovery doc; `56a9a65` DataLoader `num_workers=0` WSL2 deadlock revert; `d1255c8` WindowsPath shim for cross-OS checkpoint resume; plus HAT flash-attn, TF32/DataLoader prefetch, PSNR proxy, fp32 raster smoothing, and apex redirect fixes.
- Open issues: #14 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR backbone ablation; #8 cross-vendor kernel ports; #7 FSR4 teacher distillation; #6 v6.2-pico launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README status snapshot: pre-alpha; public positioning is still unified SR + frame extrapolation via persistent 2D Gaussian canvas / motion-vector warp / covariance resampling. v7 is the active N-D Gaussian canvas with native time axis; shipping ≤1M-param TensorRT FP8 student is not yet trained.
- Training dashboard: visible headline says `v7.0 Pico OSS-FX is training right now`, active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz step **26,000**, updated just now/live. Health tooltip contradicts this: **Trainer — offline, stuck at step 30,300 for 117h**. Watcher healthy/published 1s ago, CF Worker healthy HTTP 200 in 87ms, R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Treat as ops regression, not training progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- No newly verified arXiv paper was added today. Direct arXiv Atom API calls for `gaussian splatting` + temporal/video/dynamic/4D and `gaussian splatting` + super-resolution/arbitrary-scale hit 429/timeouts; web fallback returned no fresh arXiv hits. Do not treat this as absence of papers — rerun when arXiv recovers.
- Already-tracked high-priority leads from recent briefs remain: **Learning Video Dynamics with Predictive Differentiable Rendering** (arXiv:2606.31050), **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting** (arXiv:2606.29976), **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453), and **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.29400).

### Super-Resolution / Upscaling
- No newly verified arXiv paper was added today. Searches for real-time/game/neural SR, SR distillation/student/quantization, and temporal VSR hit arXiv API 429/timeouts and produced no web fallback hits.
- Existing relevant queue: Brane primitives / adaptive 2D-Gaussian allocation for arbitrary-scale SR, Decoder-Free Distillation for Quantized Image Restoration, and Fast/Outlier-Aware LiDAR SR as a robust-update analogue.

### Frame Extrapolation / Generation
- No newly verified arXiv paper was added today. `frame extrapolation` + neural/rendering/frame-generation arXiv query hit 429. Existing priority comparisons remain ANVIL, GFFE, Predictive Differentiable Rendering, and Flow Splatting.

### Inference Optimization
- No newly verified arXiv paper was added today. TensorRT/INT8/kernel-fusion + SR and motion-vector + SR arXiv queries hit 429. Existing optimization queue remains fused INT8/FP4 kernel work, Decoder-Free Distillation, SEADA/mixed precision, and Sol Video Inference Engine-style cache/fusion checklists.

## Archive.org Findings
- `"gaussian splatting" super-resolution` returned 3 noisy inventory/star-list captures; no useful primary paper mirror.
- `"frame generation" "deep learning"` returned 4 broad hits: MoCha-LD OSF registration, FluidFrames older builds, Minecraft Radiance snapshot, and DLSS Swapper builds. None are architecture-changing for OSS.
- `"real-time upscaling" game` returned 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS` returned 16,604 mostly polluted hits due to `GRAPE`; first results were unrelated. For known papers, targeted Wayback CDX is still the right path.
- Targeted Wayback CDX: `https://arxiv.org/abs/2507.01959` has 200 captures from 2025-07-03 and 2025-09-05; `2606.29400` timed out; `2606.29453` and `2606.31050` returned no captures in this run.

## GitHub / Open Source
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — created 2026-06-26, pushed 2026-07-02, MIT, Python, 1 star. Static structural analysis of AMD FSR 4.1.0 neural upscaler: DXIL entrypoints, weight extraction, blob layout, strict verification, rebuild section comparison. Threat/opportunity: public FSR 4.1 reverse-engineering is accelerating; useful for OSS FSR4 teacher-distillation provenance and competitor operator/layout analysis. <https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling>
- **GiulioFede/ResonantBraneSplatting** — created 2026-06-26, pushed 2026-07-01, Python, no license, 0 stars. Implements/presents Resonant Brane Splatting for arbitrary-scale SR using higher-order Brane primitives. Opportunity: direct primitive/raster ablation candidate for v7 edge/detail recovery, but license blocks reuse until clarified. <https://github.com/GiulioFede/ResonantBraneSplatting>
- **Azusa309/MedGSSR** — created/pushed 2026-06-22, MIT, Python, 2 stars. ECCV 2026 medical image SR/3D reconstruction via hierarchical feed-forward Gaussian Splatting. Relevance is representation/architecture only; medical domain mismatch is large. <https://github.com/Azusa309/MedGSSR>
- **AdvaitPalve11/Jojo-Player** — Java video player mentioning resolution upscaling/frame generation, pushed 2026-06-18. Low relevance unless it contains a reusable interpolation implementation.
- NVIDIA/AMD/Intel/vendor pages: lightweight direct fetches succeeded for NVIDIA GeForce/news, NVIDIA developer blog, AMD GPUOpen, Intel page, and Unity blog; HTML snippets were mostly generic/JS-heavy. Unreal news returned 403. Use RSS/news leads below for freshness, then primary vendor pages for confirmation.

## News
- DLSS: Google News RSS surfaced an older headline that open-source NVIDIA NVK Vulkan driver work is receiving DLSS support in Mesa 26.2. If confirmed upstream, this matters for Linux/open-driver upscaler integration expectations.
- FSR: multiple RSS headlines report FSR 4/4.1 momentum: Valve testing FSR 4 ahead of Steam Machine launch, Proton Experimental adding FSR 4 support for RDNA 3 GPUs, AMD FSR 4.1 for older GPUs, and SDK v2.3 / FSR 4.1.1 / Ray Regeneration 1.2. Competitive threat: AMD appears to be expanding neural FSR availability across Linux/Steam/older hardware faster than OSS can ship.
- Intel/XeSS: RSS had Arc G3 handheld/PCMag-style hardware coverage and one FSR4 INT8/RDNA 3.5 leak headline; no primary XeSS technical update verified.
- Apple/Metal: RSS was broad Metal-gaming commentary and M4/Mac benchmark content; no concrete MetalFX or Apple upscaler API change verified.
- GPU market: RSS includes RTX 5090/Intel Arc Pro B70 price-capacity discussion and generic GPU-deal/AI-chip content; no direct OSS architecture impact.
- Unreal/Unity: no useful fresh upscaling-specific Unity hit. Unreal primary page blocked with 403; current relevant engine signal still comes indirectly via FSR/DLSS SDK headlines.

## Action Items / Follow-ups
- [ ] Ops: dashboard says live/training but trainer health says offline stuck at step 30,300 for **117h**; reconcile public copy, health status, and host/GPU telemetry.
- [ ] Rerun arXiv search later with small `id_list`/low-count queries after 429 cooldown; today’s paper search is source-limited.
- [ ] Inspect **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** for concrete FSR 4.1 operator/weight-layout claims relevant to issue #7.
- [ ] Clone/read **ResonantBraneSplatting** only after license clarity; if usable, isolate Brane primitive raster changes as a v7 detail-recovery ablation.
- [ ] Track Proton/Valve FSR 4 support as a competitive positioning issue for Linux/handheld adoption.

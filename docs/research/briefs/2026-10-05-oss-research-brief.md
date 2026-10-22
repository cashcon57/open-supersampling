# OSS Research Brief — 2026-10-05

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`).
- **Open issues:** #14 v6.2 pico VRAM measurement; #13 Nsight L2 profiling; #12 packed low-rank canvas layout; #10 encoder ablation; #9 student reparameterization; #8 cross-vendor kernel ports; #7 teacher-signal distillation; #6 launch v6.2-pico-002; #5 no-ML shader; #4 HAT-Tiny student. GitHub API returned these first ten open issues.
- **README snapshot:** still pre-alpha, presents v7 persistent 2D Gaussian canvas / OSS-FX as the direction; README’s measured baseline remains v5-pixel-temporal (25.703 dB PSNR / 0.1666 LPIPS, TartanAir oldtown). It states HAT-Tiny is research-only (54.5 ms at 270×480 LR on 3080 Ti); the ≤1M student is not yet trained. README/dashboard status are not fully aligned: dashboard identifies active v7.0 Pico OSS-FX.
- **Training dashboard:** the earlier same-day rendered capture says `srcnn-v7.0-pico-005`, step 36,650/100,000, loss 0.19268, ETA ~1.1d; health badge says **Trainer — offline**, “stuck at step 36650 for 378h.” Watcher healthy (published 1s ago), CF Worker HTTP 200/92ms, R2 origin HTTP 200. Dashboard also shows GPU stats unavailable / host offline or rsync failed. **Refresh attempt for this memo run failed:** browser renderer could not launch (`agent-browser` permission denied), so the captured telemetry is not independently current-verified. Treat as prior same-day observation, not evidence of training progress.

## New Papers (arXiv)
Search conducted 2026-10-05; abstracts/metadata only unless noted. Material recent surfaced results:

### Gaussian Splatting / Temporal
- **Adaptive Densification for High-Fidelity and Efficient Sparse Gaussian Splatting in Arbitrary-Scale Super-Resolution** (arXiv:2606.29400, v2 2026-09-08) — Selects per-feature expansion factors with a quadtree topology; reports ~80% fewer Gaussians, ~85% lower inference time and ~73% lower memory for ~0.1 dB quality cost. **Why it matters:** a concrete adaptive capacity/compute-allocation strategy for bounded canvas storage and #12; evaluate end-to-end latency and temporal stability before adopting. [arXiv](https://arxiv.org/abs/2606.29400)
- **Ultra-fast Neural Inference for Stochastic Gaussian Splatting Denoising** (arXiv:2609.25604, 2026-09-22) — Temporal history validation uses learned trust prediction plus dual EMA accumulation and variance-gated blending. **Why it matters:** possible temporal-history confidence/gating analogue for unreliable engine motion vectors or disocclusions; adjacent rendering denoising, not SR. [arXiv](https://arxiv.org/abs/2609.25604)
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** (arXiv:2604.18047, 2026-04-20) — Continuous-time 2D Gaussian evolution with flow-guided motion, covariance resampling and adaptive offsets; already present in `BIBLIOGRAPHY.md`, so no duplicate append. [arXiv](https://arxiv.org/abs/2604.18047)

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495, 2026-07-11; Pawlicki, Kańduła, Pucek, Dobies) — Structural reparameterization yields standard-convolution inference; 644k-param version reports 28.64 dB and 27.2 FPS on Jetson Orin NX 16GB at 25W. **Why it matters:** highly relevant small-student architecture/deployment baseline for the ≤1M student; benchmark its reparameterized blocks against proposed RepVGG/ECBSR path. [arXiv](https://arxiv.org/abs/2607.10495)

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (arXiv:2608.10423, 2026-08-11) — Sparse tiled/layered scene cache supports non-neural multi-frame extrapolation with explicit disocclusion handling. **Why it matters:** competitive/technical comparator to persistent canvas extrapolation; layered visibility and disocclusion handling merit a direct comparison against OSS-FX’s single-endpoint alpha rendering. [arXiv](https://arxiv.org/abs/2608.10423)
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489, 2026-05-12; Zhao, Gao, Cai, Liu) — Combines gradient-aware super-sampling and a compact U-Net-like temporal interpolation module for 3DGS renderings. **Why it matters:** relevant unified SR+interpolation competitor, but specialized to 3DGS inputs and uses separate temporal refinement rather than OSS’s persistent canvas. [arXiv](https://arxiv.org/abs/2605.11489)

### Inference Optimization
- **Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and Automotive Accelerators** (arXiv:2609.16085, 2026-09-14; Shin) — Reports quantization speedup can be negative and depends on CPU dot-product ISA, with numerical disagreement across CPU/accelerator boundaries. **Why it matters:** reinforces measuring each student precision/backend on target hardware; do not assume INT8 portability/speedups. Not SR-specific. [arXiv](https://arxiv.org/abs/2609.16085)

## Archive.org Findings
- Archive.org Advanced Search on 2026-10-05: `"gaussian splatting" AND super-resolution` found 3 items, all unrelated GitHub-list archives; `"frame generation" AND "deep learning"` found 3 mixed results (FluidFrames builds, DLSS Swapper builds, unrelated video-generation registration); `"real-time upscaling" AND game` found 0. These noisy catalog results do **not** establish preservation of any paper.
- Exact paper-name archive checks for GS-STVSR, GSASR, GRAPE, and STSS: no new confirmed paper capture found in this run; prior dated repo briefs note historical GSASR/GRAPE captures, and acronym searches are noisy. Archive status for the others remains unverified; no takedown inference.

## GitHub / Open Source
- GitHub repository search surfaced **SplatSuRe** (96 stars, updated June 16; CVPR 2026 selective super-resolution for multi-view-consistent 3DGS), **GSASR** (102 stars; last updated Aug 6, 2025), and **GaussianSR** (111 stars; last updated Dec 27, 2024). These are existing research implementations, not a newly observed real-time game upscaler or direct temporal-canvas competitor. [SplatSuRe](https://github.com/pranav-asthana/SplatSuRe) · [GSASR](https://github.com/ChrisDud0257/gsasr) · [GaussianSR](https://github.com/tljxyys/GaussianSR)
- No clearly new open-source DLSS/FSR analysis tool was verified in this search. Search coverage is not exhaustive; GitHub web search results are discovery leads, not a complete inventory.

## News
- **NVIDIA (primary developer-blog search result, Sep 22):** DLSS 5 announcement describes 3D-guided neural rendering and controls; NVIDIA’s DLSS tag also lists DLSS 4.5 Super Resolution upgrades, Dynamic Multi Frame Generation and 6X MFG. Competitive threat is the widening vendor feature surface, but DLSS 5’s learned lighting/material detail is not directly equivalent to OSS SR/FG. [NVIDIA DLSS topic](https://developer.nvidia.com/blog/tag/dlss/)
- **AMD (GPUOpen primary site):** FSR SDK 2.3 / Redstone listing advertises FSR Upscaling 4.1.1 for RX 7000, Frame Generation 4.0.1, Ray Regeneration 1.2; the FSR UE plugin listing references UE 5.8. This further narrows the vendor-neutral positioning window if SDKs remain GPU-family-bound. [GPUOpen](https://gpuopen.com/)
- **Last 7–14 day check:** search results were sparse and mostly vendor or third-party headlines; no confidently primary-confirmed fresh Intel XeSS or Apple Metal upscaling change surfaced. Intel page extraction did not yield usable XeSS release details. Treat absence as limited search coverage, not proof no updates occurred.

## Action Items / Follow-ups
- [ ] **Ops:** investigate why v7.0-pico-005 trainer remains offline at step 36,650 for 378h while watcher and origin are healthy; dashboard GPU stats unavailable.
- [ ] **Architecture:** review adaptive sparse Gaussian densification (arXiv:2606.29400) against issue #12 packed-canvas layout; quantify temporal stability and actual runtime, not only primitive count.
- [ ] **Student:** benchmark NanoVSR-style structural reparameterization and its 644k parameter budget against issue #9 student-backbone candidates on RTX 3080 Ti.
- [ ] **FX:** compare Amulet’s explicit layered disocclusion approach to OSS-FX alpha extrapolation and document what inputs OSS can access without extra engine latency.
- [ ] **Positioning:** track DLSS 5 / DLSS 4.5 and AMD FSR Redstone capabilities and hardware constraints; maintain vendor-neutral integration as a differentiator.

*Sources: local repo git/README/BIBLIOGRAPHY; GitHub Issues API; rendered opensupersampling.com dashboard; arXiv abstract pages; GitHub repository search; Archive.org Advanced Search; NVIDIA Developer Blog and AMD GPUOpen. Search date: 2026-10-05.*

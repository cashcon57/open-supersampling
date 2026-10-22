# OSS Research Brief — 2026-08-12

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (2026-06-08); the local checkout has no newer commits. `BIBLIOGRAPHY.md` and the 2026-08-11 brief were already locally modified/untracked before this run and were preserved.
- **Open issues:** 12. The immediate performance path is #12 packed low-rank canvas layout → #13 Nsight L2 profile → #14 measured v6.2 VRAM. Shipping-model work remains #4 student distillation, #7 FSR 4 teacher-signal pipeline, #8 cross-vendor ports, #9 reparameterized student, and #10 G-buffer encoder ablation.
- **README snapshot:** v7 is a research-teacher N-D `(x,y,t)` Gaussian canvas with `V_xt` motion correlation; the ≤1M-parameter shipping CNN student and cross-vendor kernels do not yet exist.
- **Training dashboard:** `srcnn-v7.0-pico-005` displays **36,650 / 100,000**, loss **0.19268**, ETA ~1.1 days, but its trainer is **offline/stuck for 74h**. Watcher, CF Worker, and R2 origin are healthy; GPU stats are unavailable. The page simultaneously calls the run “live,” an operational state contradiction rather than evidence of progress. Best measured OSS result remains v5 pixel-temporal: 25.70 dB PSNR / 0.1666 LPIPS on held-out TartanAir oldtown. [Dashboard](https://opensupersampling.com)

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](https://arxiv.org/abs/2603.13783)** — Xuezhen Wang, Li Ma, Yulin Shen, Zeyu Wang, Pedro V. Sander; submitted 2026-03-14, revised 2026-05-06; arXiv:2603.13783, CVPR 2026. It addresses temporal aliasing in 4DGS with explicit temporal Gaussian behavior, flow-guided initialization/supervision, and triple-rendering supervision for arbitrary timestamps. **Why it matters:** unusually close validation of v7's native-time premise; deep-read its anti-ghosting supervision before altering `V_xt` or α-time curriculum.
- **[CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution](https://arxiv.org/abs/2608.02206)** — Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03; arXiv:2608.02206. Single-stage Gaussian SR jointly optimizes trusted LR observations with external HR priors, routing high-frequency gradients and densification by estimated evidence/reliability. **Why it matters:** a useful confidence-weighted densification/teacher-mixing reference, though it is sparse-view 3D reconstruction rather than streaming 2D game SR.
- **[TSOG: A Format For Temporally And Spatially Ordered Gaussians](https://arxiv.org/abs/2607.28049)** — Shady Gmira, Evangelos Alexiou, Emmanouil Potetsianakis, Emmanuel Thomas; 2026-07-30; arXiv:2607.28049. A model-agnostic 4DGS representation encodes index-aligned temporal/spatial attributes as images, reporting >90% smaller files with -0.42 to +0.85 dB PSNR differences. **Why it matters:** direct candidate for long-lived v7 canvas serialization, cache locality, and capture/upload bandwidth control.
- **[Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging](https://arxiv.org/abs/2608.10712)** — Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss; 2026-08-11; arXiv:2608.10712. Backbone-agnostic saliency-guided clustering/merging reduces redundant feed-forward Gaussian sets to roughly one-twentieth of their primitive count with controllable LoD. **Why it matters:** candidate post-spawn compaction policy for bounded canvas count; validate against temporal identity before adopting.

### Super-Resolution / Upscaling
- **[Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content](https://arxiv.org/abs/2602.11339)** — Evgeney Bogatyrev, Khaled Abud, Ivan Molodetskikh, Nikita Alutis, Dmitriy Vatolin; 2026-02-11 (revised 2026-02-14); arXiv:2602.11339. StreamSR benchmarks 11 real-time SR models on compressed YouTube-derived content and releases EfRLFN plus code/data. **Why it matters:** a useful external efficiency/robustness baseline and dataset-design reference; it does not validate the game-G-buffer or <2 ms OSS target.
- **[MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling](https://arxiv.org/abs/2608.08553)** — Rong Fu et al.; 2026-08-09; arXiv:2608.08553. A streaming VSR design combines motion fusion, sparse attention, latent-state prediction, compact decoding, and explicit temporal-smoothness/fidelity controls. **Why it matters:** a current VSR comparator and distillation-design input, but the abstract provides no sub-2 ms claim.
- **[CLEAR](https://arxiv.org/abs/2608.02206)** is also the most relevant newly found Gaussian-SR training paper; its reliability gate is more actionable than generic recent TensorRT/INT8 search hits, which were not game-SR-specific enough to add.

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423. A non-neural sparse layered image-space representation with predictive tile refresh reports up to 250 Hz at 4K. **Why it matters:** strong systems-level validation for cached extrapolation/render-rate decoupling; compare its disocclusion and refresh policy to OSS-FX, not its quality directly.
- **[Multiple Hypothesis Flow Estimation for Video Frame Interpolation under Matching Ambiguity](https://arxiv.org/abs/2608.07120)** — Zibo Su, Jing Kong, Ruixing Wang, Zhanhe Zhang, Kun Wei; 2026-08-07; arXiv:2608.07120. Reliability-routed top-K correspondences address ambiguous matches. **Why it matters:** possible conservative policy for disocclusion or invalid-MV regions, not a replacement for exact engine motion.

### Inference Optimization
- **[DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization](https://arxiv.org/abs/2608.02099)** — Minnan Pei et al.; 2026-08-03; arXiv:2608.02099 (MICRO 2026). It decouples alpha/transmittance tests from blending, compacts irregular workloads before parallel blending, and reports 1.82–6.02× end-to-end speedup over compared 3DGS accelerators. **Why it matters:** the clearest new source for #3/#12 tile-bin/raster dataflow work; translate the parsing/reorganization idea into CUDA/HIP kernels before considering accelerator-specific claims.

## Archive.org Findings
- Archive.org item search is mostly acronym noise: `GSASR` returned an unrelated CIA record; `GRAPE` returned 16,675 mostly unrelated media/viticulture items; `STSS` returned unrelated aerospace/medical/audio records. The broad Gaussian-SR query returned only secondary GitHub/awesome-list crawls.
- Targeted Wayback material exists for **[GSASR](https://web.archive.org/web/20250114052301/https://arxiv.org/abs/2501.06838)** (eight arXiv captures), **[STSS](https://web.archive.org/web/20231219050628/https://arxiv.org/abs/2312.10890)** (five arXiv captures), and **[GRAPE](https://web.archive.org/web/20260415034647/https://mulkkog.github.io/GRAPE_Page/)** (project and WACV captures).
- No qualifying Wayback capture was found for GS-STVSR’s arXiv page, and Archive.org exact search for `GS-STVSR` returned zero items. Its live [arXiv record](https://arxiv.org/abs/2604.18047) remains available; absence of a capture is not absence of the paper.

## GitHub / Open Source
- GitHub REST searches (repositories pushed since 2026-07-29) found no verified new, game-focused open Gaussian-SR implementation and no material open DLSS/FSR analysis tool. This is limited to the inspected queries, not a global absence claim.
- The verified current open baselines remain outside the window: [AMD FidelityFX SDK 2.3.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.3.0), released 2026-06-24, includes ML FSR Upscaling 4.1.1, ML Frame Generation 4.0.1, and Ray Regeneration 1.2.0; [Intel XeSS 3.0.2](https://github.com/intel/xess/releases/tag/v3.0.2), released 2026-07-24, fixes XeLL/Streamline and non-Intel-GPU issues. Neither is an August code release.

## News
- **Intel / XeSS:** Intel’s [2026-08-10 newsroom post](https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships) says planned launch integrations in *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis* use the full XeSS 3 suite: Super Resolution, Multi-Frame Generation, and Xe Low Latency. **Threat / integration signal:** evaluate combined SR+MFG+latency behavior; do not model XeSS SR in isolation.
- **NVIDIA / DLSS:** no primary NVIDIA DLSS update in this 14-day window was verified. [DLSS 4.5](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-ray-reconstruction-1000-rtx-games-apps-out-now/) remains the out-of-window competitive baseline, not an August release.
- **AMD / FSR:** no primary August SDK release was verified. Treat FSR SDK 2.3 as the current baseline rather than presenting it as new news.
- **Apple / Metal:** no qualifying Apple primary upscaling/frame-generation announcement was found in-window. No new platform requirement is established by this sweep.

## Action Items / Follow-ups
- [ ] **Critical:** restore or deliberately stop `srcnn-v7.0-pico-005`; the dashboard’s step/loss are unchanged while its trainer has been offline for 74h.
- [ ] Deep-read DeGS; prototype a tile-binning experiment that separates coverage/termination classification from blending, then measure against #12/#13 rather than extrapolating ASIC results.
- [ ] Evaluate TSOG-style index ordering/attribute packing for canvas checkpointing and capture telemetry; benchmark decode/bandwidth against the planned packed low-rank layout.
- [ ] Test saliency/reliability-gated compaction (Compact Feed-Forward / CLEAR) only with a temporal identity and disocclusion acceptance gate.
- [ ] Compare Amulet’s layered-cache refresh strategy with OSS-FX only where exact engine MVs fail or disocclusions require a refresh; retain the one-canvas architecture as the differentiator.
- [ ] Track XeSS 3’s combined SR+MFG+latency integration surface; it is a near-term ecosystem pressure point for the DLL-shim plan.

## Sources
- [OSS repository](https://github.com/cashcon57/open-supersampling) · [issues API](https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open)
- [OSS training dashboard](https://opensupersampling.com)
- Primary arXiv records: [RetimeGS](https://arxiv.org/abs/2603.13783), [EfRLFN/StreamSR](https://arxiv.org/abs/2602.11339), [CLEAR](https://arxiv.org/abs/2608.02206), [TSOG](https://arxiv.org/abs/2607.28049), [Compact Feed-Forward](https://arxiv.org/abs/2608.10712), [MotionCraft](https://arxiv.org/abs/2608.08553), [Amulet](https://arxiv.org/abs/2608.10423), [DeGS](https://arxiv.org/abs/2608.02099)
- [Intel XeSS 3 game-integration announcement](https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships)

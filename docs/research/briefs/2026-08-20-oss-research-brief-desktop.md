# OSS Research Brief — 2026-08-20

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` added tailnet-gated training controls and dashboard UI. The immediately preceding sequence preserved the Pico-005 step-5K snapshot and added cross-OS checkpoint resume, FlashAttention-2 fallback, TF32/prefetch, a corrected live-PSNR proxy, and BF16-safe FP32 smoothing.[3]
- **Open issues:** 9 open issues. The active architecture/performance queue is student distillation (#4), cross-vendor kernels (#8), RepVGG/ECBSR/RepSR student ablation (#9), decoupled G-buffer ablation (#10), packed low-rank canvas plus VRAM/Nsight work (#12–#14), and v6.2 launch/teacher-signal items (#6–#7).[2]
- **Training dashboard:** headline labels `srcnn-v7.0-pico-005` “live” at **36,650 / 100,000**, loss **0.19268**, viz **36K**, and ETA ~1.1d.[1] Service health instead says the trainer is **offline, stuck at 36,650 for 267h**; watcher, CF Worker, and R2 origin are healthy.[1] Step/loss have not advanced since the prior brief (which already reported step 36,650), so the live/ETA card is stale. GPU stats are unavailable.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **Depth Anything V4: Dynamic 4D Scene Reconstruction via Riemannian Flow Matching on 4D Gaussian Splatting** — Jiaming Fan, Jian Lu, Jinling Jia, et al.; 2026-08-18; arXiv:2608.18388. It applies flow matching over 4D-GS parameter geometry for dynamic reconstruction.[7] **Why it matters to OSS:** a teacher-side alternative to hand-authored temporal-Gaussian update rules; deep-read its parameterization and losses as an ablation for v7 `V_xt` motion, update confidence, and disocclusion handling—not as a ≤2ms student design.
- **QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting** — Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, et al.; 2026-08-18; arXiv:2608.18285. Uses anchored quantization/residual coding for dynamic Gaussian streams.[8] **Why it matters to OSS:** direct reference for #12’s packed low-rank canvas: compare anchor/residual organization, temporal deltas, and decode bandwidth before fixing a serialization format. Already present in `BIBLIOGRAPHY.md`; no duplicate added.

### Super-Resolution / Upscaling

- **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling** — Rong Fu, Chunlei Meng, Yangchen Zeng, et al.; 2026-08-09; arXiv:2608.08553. A sparse-attention latent-world-model approach to visual/video upscaling.[9] **Why it matters to OSS:** potentially useful only as a high-capacity teacher/pseudo-target reference for temporal routing and long-range consistency; it is not a plausible runtime student.
- No newly submitted direct 2D-Gaussian arbitrary-scale SR paper cleared the architecture-impact filter. `CLEAR` remains the most recent direct density-allocation lead and is already tracked.[unverified]

### Frame Extrapolation / Generation

- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, et al.; 2026-08-11; arXiv:2608.10423. Sparse layered image-space caching with adaptive shading for extrapolation.[10] **Why it matters to OSS:** strongest deterministic OSS-FX baseline/fallback for visibility and disocclusion. It remains the required comparator before claiming a single time-slice raster is sufficient for unseen surfaces; already tracked in the bibliography.

### Inference Optimization

- No material fresh TensorRT/INT8/kernel-fusion SR result passed the relevance gate. arXiv’s Gaussian-temporal and optimization API queries were HTTP 429; the one malformed broad SR/VFI response was rejected as irrelevant rather than treated as a negative result.[unverified]
- Keep already-tracked **RoofGS** as the actionable optimization reference: collect per-stage arithmetic intensity, DRAM/L2 traffic, occupancy, and sort/bin/raster timings before committing to packing or fusion work.[unverified]

## Archive.org Findings

- Archive.org remains noisy rather than evidentiary: exact `"gaussian splatting" super-resolution` returns three archived GitHub/list artifacts, exact `GS-STVSR` returns zero, and `GSASR` resolves to an unrelated CIA record.[11]
- `"frame generation" deep learning` (225 hits) and `"real-time upscaling" game` (107 hits) were dominated by unrelated media/software. No credible taken-down or superseded copy of GS-STVSR, GSASR, GRAPE, or STSS was found. Use targeted Wayback CDX against known arXiv/project URLs next run.

## GitHub / Open Source

- **MAKO** is actively updated (136 stars) and positions itself as a Vulkan-powered Linux gaming layer for Lossless Scaling frame generation/scaling. It is a **competitive-positioning watch item**, not evidence of DLSS/FSR-class reconstruction quality; inspect its capture/interposition and presentation path for OSS’s eventual Vulkan/Linux shim.[13]
- **RDNU** is a GPL-3.0 Radeon/RDNA3-focused ML upscaler using decoupled G-buffer guidance. It is a relevant open AMD-side comparator for the #10 G-buffer ablation, but its last push was July 24; validate latency and reproducibility before treating it as a threat.[14]
- **ZeroFG** is an Apache-2.0 experimental Vulkan/mobile frame-generation POC (1 star). Monitor its interface and pacing assumptions, but it is not currently a competitive threat.[12]
- **SelectiveSR** is a zero-star, unlicensed Gaussian-splatting SR repository updated today. Its description claims multi-scale efficiency but no demonstrated evaluation/license supports adopting or positioning against it yet.[15]
- No verified new open-source DLSS/FSR binary-analysis tool surfaced. Generic GitHub search was otherwise course repos, latent-video tooling, or WIP upscalers.[5]

## News

- **RSS/headline-level only:** multiple August 10–19 reports say *Beast of Reincarnation* gained DLSS 4.5/FSR 4 and *Alan Wake 2* gained AMD FSR 4 plus frame generation.[16] This is game integration momentum, not a verified SDK/API change or benchmark result.
- No material primary NVIDIA, AMD, Intel, Apple, Unreal, or Unity upscaling/runtime announcement was verified in the past 14 days. NVIDIA/Intel/Apple landing pages were reachable; AMD’s primary page timed out. Treat any quality/performance claims in the RSS feed as unverified.[unverified]

## Action Items / Follow-ups

- [ ] **Ops blocker:** restore or explicitly stop/relabel Pico-005. It has remained at step 36,650 while trainer health now reports a 267h outage.[1]
- [ ] Deep-read Depth Anything V4 for temporal-state uncertainty/motion-loss design; reject flow-matching complexity unless a small v7 teacher ablation improves hard-motion/disocclusion metrics.[7]
- [ ] Turn QuARC-GS into a bounded #12 experiment: anchored vs flat record layout, temporal-delta bytes/Gaussian, decode bandwidth, and raster impact.[8]
- [ ] Benchmark Amulet-style layered-cache visibility/disocclusion against the native `t_query` OSS-FX render on identical motion-vector inputs.[10]
- [ ] Watch MAKO/ZeroFG and audit MAKO’s license/interposition implementation; competitive threat remains low, but Vulkan integration learnings may be reusable.[12][13]

## Sources

[1] https://opensupersampling.org
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[3] https://github.com/cashcon57/open-supersampling/commits/main
[5] https://api.github.com/search/repositories
[7] https://arxiv.org/abs/2608.18388
[8] https://arxiv.org/abs/2608.18285
[9] https://arxiv.org/abs/2608.08553
[10] https://arxiv.org/abs/2608.10423
[11] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&rows=10&output=json
[12] https://github.com/hy300leosquizz-ctrl/ZeroFG
[13] https://github.com/eugeniosegala/MAKO
[14] https://github.com/Adam-AtlasSoftware/RDNU
[15] https://github.com/gomzee1024/SelectiveSR
[16] https://news.google.com/rss/search?q=DLSS+4+OR+FSR+4+OR+XeSS+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen

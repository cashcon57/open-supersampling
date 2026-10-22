# OSS Research Brief — 2026-08-23

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`; recent work is dashboard-test/infrastructure, not a v7 model or kernel change.
- **Open issues:** 12: P0 raster/tile-bin work (#1/#3), packed-canvas and measurement gates (#12–#14), student/distillation (#4/#7/#9), shader fallback (#5), launch/ablation (#6/#10), and cross-vendor ports (#8).[1]
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is the active HAT-Tiny teacher; the ≤1M shipping student and non-CUDA kernels remain unbuilt.[4]
- **Training dashboard:** live data reports v7 at **step 36,650**, canvas count 2,304, total loss 0.19268, SR Charbonnier 0.02360, and FG Charbonnier 0.07643. The same endpoint says trainer **offline/stuck for 1h** while watcher, Worker, and R2 origin are healthy; its generated timestamp is 2026-08-24T00:15:50Z (UTC), so this is an operational stale-run signal, not evidence of model progress.[2][3]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting](https://arxiv.org/abs/2608.15785)** (Yang Luo, Yan Gong, Yongsheng Gao, Jie Zhao; 2026-08-16; arXiv:2608.15785) — Roofline-guided compact attributes, 32-bit depth keys, bounded-range fast-exp, culling, dual-pixel work, and kernel fusion target the full 3DGS pipeline. **Why it matters to OSS:** highest-value immediate read for #3/#12/#13; use its decomposition to test whether the low-rank canvas is front-end bandwidth- or raster-instruction-bound before selecting a fused kernel design.[5]
- **[S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices](https://arxiv.org/abs/2608.19639)** (Yiwei Li, Jiannong Cao, Weixun Gao, Rui Cao, Songye Zhu, Yinfeng Cao, Mingjin Zhang; 2026-08-20; arXiv:2608.19639) — octree-organized Gaussian residuals, temporal gating, and sparse updates target online dynamic-GS reconstruction. **Why it matters:** a newly material compact-update reference for v7 canvas residency; its edge deployment claims make it useful for evaluating temporal-update sparsity without adopting its free-viewpoint system wholesale.[6]
- **QuARC-GS** (Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, Kevin C. Zhou; 2026-08-18; arXiv:2608.18285) — canonical-frame residual coding, quantized deformation, and change-gated densification reduce dynamic-GS streaming storage. **Why it matters:** reinforces primitive-identity preservation and sparse updates for #12, but is already tracked in `BIBLIOGRAPHY.md` and is not a drop-in game rasterizer.[7]

### Super-Resolution / Upscaling
- **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling** (Rong Fu et al.; 2026-08-09; arXiv:2608.08553) — motion-aware latent state and sparse attention trade temporal smoothness against fidelity for VSR. **Why it matters:** teacher-side comparator for whether a persistent Gaussian state improves long-range temporal behavior; not evidence for the ≤1M / <2 ms shipping graph.[8]
- No new directly relevant TensorRT/INT8 paper passed the deployment-relevance gate. RoofGS is the material inference result this run; reproduce or reduce its claims to per-stage measurements before changing OSS latency assumptions.[5]

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423) — a sparse tiled layered image-space cache predicts future visibility/shading and selectively refreshes tiles. **Why it matters:** **competitive threat and opportunity**: it validates persistent-representation/time-query FG, but its explicit disocclusion cache is a stronger control than pure motion warping. Deep-read its visibility and refresh policy before fixing OSS-FX occlusion supervision.[9]
- **SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation** (Jisoo Jeong et al.; 2026-08-13; arXiv:2608.13460) — confidence maps fuse nonlinear-flow intermediate candidates for occlusions and boundaries. **Why it matters:** a teacher-loss/confidence-ablation lead only; diffusion inference excludes it from OSS’s shipping path.[10]

### Inference Optimization
- RoofGS is the only new material optimization paper. Treat its compact layout, key packing, and fusion sequence as independently measurable hypotheses for the current P0 profiling gates—not as directly transferable FPS.[5]

## Archive.org Findings
- Broad Archive Advanced Search was noisy: Gaussian-SR returned three archive/listing artifacts; frame-generation and real-time-game-upscaling results were predominantly unrelated media or repository snapshots. No material primary paper artifact was found.[11][12][13]
- Targeted Wayback CDX found no 200 capture for GS-STVSR (`2604.18047`) abstract/PDF. GSASR, GRAPE, and STSS acronym searches were similarly too noisy or had no verified relevant capture. This is **archive absence**, not evidence the papers were removed.[14]

## GitHub / Open Source
- **[SelectiveSR](https://github.com/gomzee1024/SelectiveSR)** was pushed 2026-08-20 and claims multi-scale Gaussian-splatting SR, but has no declared license and no adoption signal. Monitor/benchmark only; it is the closest fresh Gaussian-SR lead, not a validated competitor.[15]
- **[OSSS](https://github.com/Lenovive/OSSS)** is a new MIT C++ Windows capture → optical-flow → interpolation/upscale implementation, pushed 2026-08-22. It is useful plumbing reconnaissance for a future OSS DLL/capture path, not a quality competitor.[16]
- **[ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** is a tiny Apache-2.0 Vulkan/mobile FG proof of concept. Watch for cross-vendor pacing/integration patterns; adoption and technical depth are currently negligible.[17]
- No material new open-source DLSS/FSR analysis tool passed verification. A Decky/OptiScaler integration is a distribution signal for SteamOS, not a new reconstruction algorithm.[18]

## News
- No direct-primary-page-verified NVIDIA, AMD, Intel, Apple, Unreal, or Unity upscaling/API announcement was recovered in the 2026-08-09–23 window. Vendor extraction/search was credit-blocked.
- Google News RSS provides only headline-level leads: DLSS 4.5 title adoption, AMD Adrenalin 26.8.1 release notes, and an Intel Arc driver entry. Do not turn these into OSS architecture claims until direct vendor pages are checked.[19][20][21]

## Action Items / Follow-ups
- [ ] **P0 ops:** diagnose/restart or explain the v7 trainer at step 36,650; the dashboard’s healthy publisher/origin proves publication is working, not that training is advancing.
- [ ] **RoofGS deep read / reproduce:** map its 32-bit keys, compact attributes, fast-exp bound, and fusion sequence to #3/#12; use #13 Nsight evidence to select the bottleneck.
- [ ] **Amulet control:** prototype layered future-visibility plus adaptive tile lifetime as an OSS-FX disocclusion control before claiming fractional-time rendering has a visibility strategy.
- [ ] **S²GS ablation:** test sparse residual/temporal-gating updates against v7’s parent-child canvas while retaining temporal primitive identity and fixed L2-residency limits.
- [ ] Keep MotionCraft/SNM-VFI teacher-side only; do not import large attention or diffusion into the deployment graph.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://opensupersampling.com/status.json
[3] https://opensupersampling.com/data.json
[4] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[5] https://arxiv.org/abs/2608.15785
[6] https://arxiv.org/abs/2608.19639
[7] https://arxiv.org/abs/2608.18285
[8] https://arxiv.org/abs/2608.08553
[9] https://arxiv.org/abs/2608.10423
[10] https://arxiv.org/abs/2608.13460
[11] https://archive.org/advancedsearch.php?q=title%3A%28gaussian+splatting+super+resolution%29+OR+description%3A%28gaussian+splatting+super+resolution%29&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&fl%5B%5D=mediatype&rows=10&output=json
[12] https://archive.org/advancedsearch.php?q=title%3A%28frame+generation+deep+learning%29+OR+description%3A%28frame+generation+deep+learning%29&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&fl%5B%5D=mediatype&rows=10&output=json
[13] https://archive.org/advancedsearch.php?q=title%3A%28real+time+upscaling+game%29+OR+description%3A%28real+time+upscaling+game%29&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&fl%5B%5D=mediatype&rows=10&output=json
[14] https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Farxiv.org%2Fabs%2F2604.18047&output=json&filter=statuscode%3A200
[15] https://api.github.com/repos/gomzee1024/SelectiveSR
[16] https://api.github.com/repos/Lenovive/OSSS
[17] https://api.github.com/repos/hy300leosquizz-ctrl/ZeroFG
[18] https://api.github.com/repos/danielcamilo1/decky-optiscaler
[19] https://news.google.com/rss/search?q=site%3Anvidia.com+OR+site%3Anvidianews.nvidia.com+DLSS+after%3A2026-08-08+before%3A2026-08-24&hl=en-US&gl=US&ceid=US%3Aen
[20] https://news.google.com/rss/search?q=site%3Aamd.com+OR+site%3Agpuopen.com+FSR+Radeon+after%3A2026-08-08+before%3A2026-08-24&hl=en-US&gl=US&ceid=US%3Aen
[21] https://news.google.com/rss/search?q=site%3Aintel.com+XeSS+Arc+GPU+after%3A2026-08-08+before%3A2026-08-24&hl=en-US&gl=US&ceid=US%3Aen

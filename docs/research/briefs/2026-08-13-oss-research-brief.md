# OSS Research Brief — 2026-08-13

## Project Status Summary
- **Latest commits:** local `main` remains at `b44278a test: add local Chrome Playwright dashboard smoke`; no newer commit appears in the most recent ten.
- **Open issues:** 12. The immediate performance path remains #12 packed low-rank canvas layout → #13 Nsight L2 profile → #14 measured VRAM footprint. Student distillation (#4), FSR 4 teacher signal (#7), vendor ports (#8), reparameterized student (#9), and the G-buffer ablation (#10) remain open.
- **README snapshot:** OSS remains a research-teacher v7 N-D `(x,y,t)` Gaussian canvas with `V_xt` motion correlation. The ≤1M-parameter shipping CNN student and all non-CUDA kernels are still unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` is still **36,650 / 100,000**, loss **0.19268**, while health says **trainer offline / stuck for 98h**. Watcher, CF Worker, and R2 origin are healthy; GPU statistics are unavailable. Its simultaneous “live” label is a dashboard-state contradiction, not evidence of training progress.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution](https://arxiv.org/abs/2608.02206)** (arXiv:2608.02206; Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03) — single-stage 3DGS SR uses LR observations as a reliable anchor, evidence-conditioned high-frequency gradient correction, and patch-to-Gaussian reliability routing/densification. **Why it matters to OSS:** its explicit confidence gate is a concrete candidate for loss-adaptive spawning and for mixing trustworthy engine captures with imperfect teacher priors.[2]
- **[Resonant Brane Splatting for Arbitrary-Scale Super-Resolution](https://arxiv.org/abs/2606.29453)** (arXiv:2606.29453; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto; 2026-06-28, revised 2026-07-01) — replaces flat 2D Gaussians with Gaussian-Hermite “Branes” carrying spatially varying color, then combines fewer-overlap primitives with a culling-aware differentiable rasterizer. **Why it matters to OSS:** this is newly surfaced, direct alternative primitive design for high-frequency detail and tile workload; benchmark its quality-per-primitive before expanding v7 state.[3]
- **[Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution](https://arxiv.org/abs/2606.29400)** (arXiv:2606.29400; Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto; 2026-06-28) — QuADA-GS predicts a global-budgeted local densification allocation and uses hierarchical pointer convolution for irregular feature communication. **Why it matters to OSS:** the budgeted routing policy maps directly to #12’s bounded canvas/tile scheduler, though it needs temporal-identity validation before adoption.[4]
- **[DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization](https://arxiv.org/abs/2608.02099)** (arXiv:2608.02099; Minnan Pei, Gang Li, Zeyu Zhu, Siting Wang, Junwen Si, Zhuoran Song, Yu Feng, Fangxin Liu, Xiaoyao Liang, Jian Cheng; 2026-08-03) — separates coverage/transmittance tests from blending and compacts irregular work before parallel processing. **Why it matters to OSS:** strongest new raster dataflow reference for #3/#12 custom tile bins and fused CUDA/HIP kernels; translate the decomposition, not accelerator-specific speed claims.[5]

### Super-Resolution / Upscaling
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (arXiv:2607.10495; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11) — a fully convolutional structurally reparameterized VSR model deploys as standard convolutions, reports 28.64 dB and 27.2 FPS for its 644K variant on Jetson Orin NX, and avoids explicit flow/custom operations. **Why it matters to OSS:** best current student-backbone reference for #9/#4; the deploy graph is promising, but its reported throughput is not evidence for OSS’s <2 ms target.[6]
- **[MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling](https://arxiv.org/abs/2608.08553)** (arXiv:2608.08553; Rong Fu, Chunlei Meng, Yangchen Zeng, Xiaowen Ma, Yongtai Liu, Wangyu Wu, Shuo Yin, Zijian Zhang, Sicheng Li, Yingrui Ji, Chenhao Wang, Simon Fong; 2026-08-09) — motion-aware latent-state prediction combines motion fusion, sparse attention, a compact decoder, and a smoothness/fidelity control. **Why it matters to OSS:** a teacher-side comparator for persistent state and sparse nonlocal context, not a shipping-latency baseline.[7]

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (arXiv:2608.10423; Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11) — a tiled layered image-space cache predicts future visibility/shading, refreshes stale tiles adaptively, explicitly handles disocclusions, and reports up to 250 Hz at 4K. **Why it matters to OSS:** a credible systems-level competitive threat to OSS-FX’s cached/extrapolated-frame thesis; compare its disocclusion, tile lifetime, and refresh policy against OSS’s exact-engine-MV Gaussian canvas.[8]

### Inference Optimization
- No newly material TensorRT/INT8-only paper beat the direct deploy-graph lessons already captured in NanoVSR and DeGS. Recent generic quantization hits remain lower priority than measuring a fused FP8/INT8 student against #9/#12.

## Archive.org Findings
- Archive.org item search remains mostly noise: broad Gaussian-SR queries yielded secondary awesome-list snapshots; real-time game-upscaling yielded no useful primary artifact; exact GS-STVSR had no item/captured arXiv page. This does not affect the live paper record.
- Durable targeted captures remain useful: [GSASR arXiv](https://web.archive.org/web/20250114052301/https://arxiv.org/abs/2501.06838), [GSASR GitHub](https://web.archive.org/web/20260607093324/https://github.com/ChrisDud0257/GSASR), [STSS arXiv](https://web.archive.org/web/20231219050628/https://arxiv.org/abs/2312.10890), and [GRAPE](https://web.archive.org/web/20260415034647/https://mulkkog.github.io/GRAPE_Page/). No removed/updated OSS-relevant paper was newly recovered.[9]
- [FluidFrames/RIFE builds](https://archive.org/details/fluidframeswin) and [DLSS Swapper historical builds](https://archive.org/details/dlssswapperbuildswin) are useful deployment/shim-history artifacts, not model-quality baselines.

## GitHub / Open Source
- GitHub REST searches for repositories pushed since 2026-07-30 found no verified game-focused Gaussian-SR implementation or new DLSS/FSR/XeSS analysis tool. The hits were unrelated video-generation projects; `julesklord/bananascaler` is a zero-star MIT WIP UI, not a competitor.[10]
- NanoVSR’s official [MIT-licensed code](https://github.com/filippawlicki/nanovsr) is the only material new code lead (35 stars; last push 2026-08-04). Pull it only for deploy-graph/benchmark inspection, not as an unmeasured architectural transplant.[6]
- Existing ecosystem baselines remain [FidelityFX SDK 2.3.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.3.0), released 2026-06-24, and [XeSS SDK 3.0.2](https://github.com/intel/xess/releases/tag/v3.0.2), released 2026-07-24; neither is an August code release.[11][12]

## News
- **NVIDIA / DLSS:** NVIDIA’s 2026-08-04 *Gears of War: E-Day* beta post verifies UE5 + DX12 and DLSS Super Resolution for the beta; it says the October release will support the full DLSS 4.5 suite. Treat this as an integration signal, not a beta claim for the whole FG/reconstruction stack.[13]
- **Intel / XeSS:** Intel’s 2026-08-10 partnership announcement says *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis* will ship with the complete XeSS 3 suite—SR, Multi-Frame Generation, and Xe Low Latency. **Threat:** Pico’s segment is receiving a bundled SR+FG+latency stack; assess integrated latency and cross-vendor fallback rather than SR in isolation.[14]
- **AMD / FSR, Apple / Metal, engines:** no qualifying primary AMD FSR 4, Apple MetalFX, or engine-level Unreal/Unity upscaling announcement was verified in the reviewed 14-day window. Unity 6000.5.8f1’s published notes did not mention these features. This is a limited source-sweep result, not proof of global absence.

## Action Items / Follow-ups
- [ ] **Critical:** restore or deliberately terminate `srcnn-v7.0-pico-005`; the dashboard step/loss are unchanged and trainer-offline duration is now 98h.
- [ ] Deep-read Amulet; create a bounded OSS-FX ablation for disocclusions/invalid engine MVs comparing layered refresh cost, holes, and latency.
- [ ] Deep-read QuADA-GS and CLEAR; design an engine-MV-aware confidence gate for spawning/densification with a temporal identity acceptance test.
- [ ] Benchmark a NanoVSR-style reparameterized standard-conv student against #9’s RepVGG/ECBSR/RepSR candidates; test post-fusion FP8/INT8 TensorRT before selecting a backbone.
- [ ] Translate DeGS’s parse/compact/blend split into a #12 prototype and profile it with #13; do not extrapolate ASIC speedups.
- [ ] Treat DLSS 4.5, FSR 4, and XeSS 3 as integrated SR+FG+latency competitors and establish a shared capture protocol before making quality/latency positioning claims.

## Sources
[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.02206
[3] https://arxiv.org/abs/2606.29453
[4] https://arxiv.org/abs/2606.29400
[5] https://arxiv.org/abs/2608.02099
[6] https://arxiv.org/abs/2607.10495
[7] https://arxiv.org/abs/2608.08553
[8] https://arxiv.org/abs/2608.10423
[9] https://archive.org/advancedsearch.php
[10] https://api.github.com/search/repositories
[11] https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.3.0
[12] https://github.com/intel/xess/releases/tag/v3.0.2
[13] https://www.nvidia.com/en-us/geforce/news/gears-of-war-e-day-multiplayer-beta-august-6-dlss/
[14] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships

# OSS Research Brief — 2026-08-16

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke` (only reachable recent commit in this checkout).
- **Open issues:** 12. Critical dependency chain remains #12 packed low-rank canvas layout → #13 Nsight L2 profile → #14 v6.2 VRAM measurement; student distillation (#4/#9), FSR 4 teacher signal (#7), cross-vendor kernels (#8), and the G-buffer ablation (#10) remain open.
- **README snapshot:** v7 is the active N-D `(x,y,t)` Gaussian teacher with `V_xt` cross-correlation. The ≤1M shipping student and non-CUDA backends remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` still shows **36,650 / 100,000**, loss **0.19268**, but Trainer is **offline/stuck for 170h** and GPU telemetry is unavailable. Watcher, Worker, and R2 are healthy. The “live” label is contradicted by the frozen metric state, not evidence of progress.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting](https://arxiv.org/abs/2604.18047)** (arXiv:2604.18047; Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20; already tracked) — continuous 2D-Gaussian VSR with motion alignment and covariance resampling. **Why it matters:** closest direct temporal-SR comparator for v7’s raster-space covariance behavior; benchmark disocclusion stability and primitive lifetime separately from observed-pair interpolation.
- **[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](https://arxiv.org/abs/2603.13783)** (arXiv:2603.13783; Xuezhen Wang, Li Ma, Yulin Shen, Zeyu Wang, Pedro V. Sander; 2026-03-14; already tracked) — models continuous-time Gaussian behavior with flow-guided and triple-rendering supervision. **Why it matters:** strongest external control for v7’s native time-axis thesis; retain engine MVs and test its time-supervision/occlusion ablations.

### Super-Resolution / Upscaling
- **[Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training](https://arxiv.org/abs/2604.20291)** (arXiv:2604.20291; Pham Phuong Nam Nguyen, Nam Tien Le, Thi Kim Trang Vo, Nhu Tinh Anh Nguyen; 2026-04-22; newly material) — keeps compute at LR, uses a re-parameterizable backbone plus PixelShuffle, then applies QAT on the fused deploy graph with teacher-guided supervision. **Why it matters:** a concrete sequence for the OSS ≤1M student: fuse first, then QAT/calibration; its x3/mobile SISR result is not a game-VSR latency claim.[2]
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (arXiv:2607.10495; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; already tracked) — 644K fully-convolutional VSR structurally reparameterized to standard convolutions. **Why it matters:** strongest deploy-graph control for #4/#9; benchmark at OSS resolutions with exact-MV warp and a tiny confidence head.
- **[DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](https://arxiv.org/abs/2603.22271)** (arXiv:2603.22271; Zhengyao Lv, Menghan Xia, Xintao Wang, Kwan-Yee K. Wong; 2026-03-23; already tracked) — one-step diffusion-VSR distillation combines trajectory-preserving initialization, distribution matching, adversarial score features, and preference refinement. **Why it matters:** teacher-side perceptual-distillation reference only, not a viable shipping graph.

### Frame Extrapolation / Generation
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** (arXiv:2605.11489; Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12; newly material) — post-render GASS super sampling and a compact U-Net/GRU temporal interpolation module improve rendered 3DGS outputs. **Why it matters / threat:** it validates the joint-SR-plus-temporal-rendering product surface, but still uses separate refinement/interpolation modules; OSS must prove the stronger shared-canvas, fractional-time claim at comparable quality and latency.[3]
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (arXiv:2608.10423; Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; already tracked) — sparse layered image-space cache, predictive shading, and adaptive tile refresh for extrapolation. **Why it matters / threat:** use its layer/disocclusion and refresh policy as OSS-FX controls; exact engine MVs remain OSS’s differentiation.

### Inference Optimization
- **Efficient INT8 SISR** above is the only newly material primary deployment paper this pass. Its fused-deploy-graph QAT, clipping, and BatchNorm recalibration are actionable after #12/#13/#14 establish the packed-state/raster bandwidth budget.[2]
- No new primary TensorRT or kernel-fusion paper cleared the materiality gate. The decision order remains: implement #12, profile #13, measure #14, then select fusion and quantization work from occupancy/bandwidth data.

## Archive.org Findings
- Archive.org advanced-search responses were non-JSON at query time; web-index fallback exposed only generic/deep-learning artifacts, not a verified GS-STVSR/GSASR/GRAPE/STSS primary artifact.[4]
- Targeted Wayback CDX found historical captures for the older GaussianSR abstract (`2407.18046`), but no verified archived primary capture for GS-STVSR or GSASR in this pass.[5]

## GitHub / Open Source
- **Threat / integration bar — [OptiScaler](https://github.com/optiscaler/OptiScaler)** (GPL-3.0; 9,822 stars; pushed 2026-08-16) bridges DLSS2+/XeSS/FSR2+ inputs and enables FSR-FG/XeFG on non-FG titles. This is the practical cross-vendor compatibility benchmark for the future DLL shim—not a license-compatible dependency.[6]
- **Opportunity — [RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** (GPL-3.0; 15 stars; updated 2026-08-11) is an early RDNA3-focused G-buffer-guided neural upscaler. Mine validation/kernel ideas only; GPL and low maturity rule out direct reuse.[7]
- Requested GitHub searches produced no newly pushed material Gaussian-SR, open-FG, neural-upscaler, or DLSS/FSR-analysis implementation. `julesklord/bananascaler` is an MIT but zero-star WIP TUI wrapper, not competitive technical signal.[12]
- **Vendor bar:** AMD FSR SDK 2.3 ships FSR Upscaling 4.1.1, Frame Generation 4.0.1, and Ray Regeneration 1.2.0; eligible DX12 games with FSR 3.1+ can receive driver-provided ML updates on RX 7000/9000 hardware. This raises integration/maintenance expectations for OSS’s DLL path.[8]

## News
- **2026-08-14 — MSFS 2024 Sim Update 6:** final release reverted FSR 4 to FSR 3 because FSR 4 frame generation was incompatible with multiple rendering windows. **OSS implication:** make multi-window rendering and fallback behavior first-class in the FG compatibility matrix.[9]
- **2026-08-13 — NVIDIA title integrations:** NVIDIA confirmed DLSS 4.5 for *Gears of War: E-Day*, *Hell Let Loose: Vietnam*, *No More Room In Hell 2*, and *Phantom Blade Zero*. Competitive pressure is title coverage, not a new OSS-relevant API.[10]
- Intel’s current Arc G-series material positions XeSS 3 as SR + multi-frame generation + low-latency for handhelds. It is not a fresh 7–14-day announcement, but reinforces that OSS Pico needs handheld-class measurements rather than desktop TensorRT extrapolation.[11]
- RSS-only reports of FSR 4.1 performance costs and a DLSS 5 claim were not promoted: no primary confirmation was found.

## Action Items / Follow-ups
- [ ] **Critical:** inspect/restart `srcnn-v7.0-pico-005`; frozen step/loss, null GPU telemetry, and 170h trainer-offline state override its “live” label.
- [ ] Implement the #12 → #13 → #14 measurement chain before committing to any fused/quantized student path.
- [ ] Prototype re-parameterized LR-domain student blocks, then QAT the fused graph with clipping + BN recalibration; compare against #9’s RepVGG/ECBSR/RepSR candidates.
- [ ] Add 3DGS³ as a product-surface comparator: report whether OSS fractional-time canvas rendering can match a two-module GASS+interpolation baseline under a shared quality/latency budget.
- [ ] Build the DLL-shim compatibility matrix around OptiScaler/FSR/XeSS inputs; explicitly include multi-window rendering, swap-chain lifetime, and GPL boundaries.

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2604.20291
[3] https://arxiv.org/abs/2605.11489
[4] https://archive.org/advancedsearch.php?q=%22gaussian%20splatting%22%20super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&output=json&rows=8
[5] https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Farxiv.org%2Fabs%2F2407.18046&output=json&filter=statuscode%3A200
[6] https://github.com/optiscaler/OptiScaler/releases/tag/v0.9.4
[7] https://github.com/Adam-AtlasSoftware/RDNU
[8] https://gpuopen.com/amd-fsr-sdk
[9] https://www.flightsimulator.com/release-notes-sim-update-6-1-8-14-0-now-available-msfs-2024
[10] https://www.nvidia.com/en-us/geforce/news/no-more-room-in-hell-2-hell-let-loose-vietnam-dlss
[11] https://newsroom.intel.com/client-computing/intel-arc-g-series-processors-set-a-new-standard-for-handheld-pc-gaming
[12] https://api.github.com/repos/julesklord/bananascaler

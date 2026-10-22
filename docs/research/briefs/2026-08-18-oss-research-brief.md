# OSS Research Brief — 2026-08-18

## Project Status Summary
- **Latest commits:** this shallow checkout still exposes only `b44278a test: add local Chrome Playwright dashboard smoke`. The working tree already had the 2026-08-11–17 briefs and a pre-existing `BIBLIOGRAPHY.md` research update.[unverified]
- **Open issues:** 12. The P0 performance dependency remains **#12 packed low-rank canvas → #13 Nsight L2 profile → #14 VRAM measurement**; #1 low-rank rasterizer and #3 counting-sort tile bins are also P0. Student distillation (#4/#7/#9), non-CUDA ports (#8), and G-buffer ablation (#10) remain open.[9]
- **README snapshot:** v7 is the active teacher architecture: native `(x,y,t)` Gaussians with `V_xt` cross-correlation. The ≤1M shipping student, DLL shim, and HIP/Metal/Vulkan/Level Zero kernels are still unbuilt.[10]
- **Training dashboard:** `srcnn-v7.0-pico-005` remains **offline at step 36,650**, now reported stuck for **218h**; watcher, Cloudflare Worker, and R2 are healthy.[1] That is +12h without progress since the prior brief.[1] The stale dashboard v6.1 label versus the v7 active-run field remains an operational data-quality defect, not evidence of recovery.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives](https://arxiv.org/abs/2412.20720)** (arXiv:2412.20720; Zeyu Yang, Zijie Pan, Xiatian Zhu, Li Zhang, Jianfeng Feng, Yu-Gang Jiang, Philip H. S. Torr; 2024-12-30, revised 2025-08-05; newly added to bibliography) — optimizes native 4D Gaussians with a time-conditioned rendering pipeline and compact variants. **Why it matters:** **HIGH RELEVANCE** independent precedent for v7's native time axis and covariance cross-terms; compare its memory controls against the packed-canvas plan before expanding primitive state.[16]
- **[ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields](https://arxiv.org/abs/2608.04581)** (arXiv:2608.04581; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko; 2026-08-05; ACM MM 2026) — assigns Gaussian cardinality and feature-channel width by local spatiotemporal complexity. **Why it matters:** clean ablation source for bounded parent/child spawning under #12, but gate it on temporal identity and measured L2 residency rather than paper compression alone.[2]
- **[GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors](https://arxiv.org/abs/2608.13502)** (arXiv:2608.13502; Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang; 2026-08-13; already tracked) — opacity-guided splitting, mean-opacity regulation, and explicit candidate/count caps stabilize densification under ambiguous observations. **Why it matters:** candidate safety controls for disocclusion spawning and bad engine-MV/capture inputs; domain is not game SR.[11]

### Super-Resolution / Upscaling
- **[MeanSR: Restoration Trajectory Learning for One-Step Perceptual Super-Resolution](https://arxiv.org/abs/2608.09405)** (arXiv:2608.09405; Axi Niu, Jiawei Kou, Kang Zhang, Qingsen Yan, Jinqiu Sun, Yanning Zhang; 2026-08-10; already tracked) — learns an LR-conditioned average restoration-velocity field for one-step perceptual SR, avoiding an expensive pretrained teacher. **Why it matters:** useful teacher-free control for the HAT→≤1M student, but it provides no game-temporal, exact-MV, or sub-2ms evidence.[3]
- **[MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling](https://arxiv.org/abs/2608.08553)** (arXiv:2608.08553; Rong Fu et al.; 2026-08-09; already tracked) — motion-aware latent state plus sparse attention targets controllable temporal VSR. **Why it matters:** teacher-side comparator for persistent state and smoothness controls, not a shipping-graph candidate.[12]
- No fresh material TensorRT/INT8/kernel-fusion paper appeared in this scan; deployment choice should remain blocked on #12→#13→#14 measurements.[unverified]

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (arXiv:2608.10423; Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; already tracked) — a layered image-space cache pre-fills future visibility/shading and adaptively refreshes stale tiles; it reports extrapolation from 60Hz shading toward 240Hz display. **Why it matters / threat:** the most direct systems competitor to OSS-FX on disocclusion and refresh policy, though it does not share OSS’s exact-engine-MV Gaussian-canvas premise.[4]
- **[SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation](https://arxiv.org/abs/2608.13460)** (arXiv:2608.13460; Jisoo Jeong, Hong Cai, Jamie Menjay Lin, Hanno Ackermann, Hyeonjun Sim, Yinhao Zhu, Yunxiao Shi, Fatih Porikli; 2026-08-13; already tracked) — combines nonlinear flow candidates with confidence-guided diffusion detail recovery. **Why it matters:** not latency-compatible, but a useful invalid-MV/occlusion confidence control.[13]
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** remains the closest product-surface comparator: it couples SR and interpolation, but retains separate modules. OSS must eventually publish a shared content/latency bake-off to support the stronger single-canvas claim.[14]

### Inference Optimization
- No fresh primary TensorRT/INT8/kernel-fusion paper was verified: the arXiv Atom API returned HTTP 429/timeouts for the targeted queries. Prioritize measured packed layout, L2 behavior, then VRAM footprint before committing to fused kernels or quantization. ACA-GS supplies capacity-allocation ideas; it does not validate the 32B/Gaussian target.[2]

## Archive.org Findings
- `gaussian splatting super-resolution` returned only three GitHub snapshot items, not a preserved paper artifact; `frame generation deep learning` and `real-time upscaling game` were largely generic/noisy inventory. No usable archive copy or removed-paper recovery surfaced.[5]
- Targeted Wayback checks found 200 captures for QuADA-GS (2026-07-02), RBS (2026-07-02), and 3DGS³ (2026-06-07); no 2026 capture was returned for GS-STVSR. Treat this as incomplete archive coverage, not absence of the paper.[15]

## GitHub / Open Source
- **[ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** — newly public Apache-2.0 Vulkan frame-generation V1 POC (13 source/shader files; last commit Aug. 13) accepts host Vulkan resources/command buffers and performs block-motion, confidence, midpoint warp/blend. **Threat level: low; opportunity: medium.** It is interpolation-only and has no independently validated build system, but its small explicit host contract is worth reviewing for OSS's eventual Vulkan integration seam.[17]
- **[RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** — a GPL-3.0 RDNA3-focused decoupled-G-buffer neural upscaler with FP16/INT8 artifacts and CMake/runtime scaffolding. **Opportunity:** inspect its G-buffer contract and quantized deployment layout as a cross-vendor implementation reference; do not reuse GPL code in Apache-2.0 OSS.[18]
- GitHub REST search otherwise produced no directly relevant new Gaussian-splatting/SR repository; broad hits were dominated by unrelated video-generation tools. No qualifying new open DLSS/FSR-analysis tool was found.[6]

## News
- **NVIDIA, Aug. 13:** Gears of War: E-Day’s full release is stated to include the DLSS 4.5 suite, while No More Room In Hell 2 and Hell Let Loose: Vietnam launch with DLSS 4.5.[7] **Positioning:** title support bundles SR, frame generation, Reflex, and RT features; OSS must report SR/FG/runtime support independently rather than as one generic “upscaler” checkbox.
- **Intel, Aug. 10:** Intel announced launch-day XeSS 3 Super Resolution, Multi-frame Generation, and Xe Low Latency enablement for STAR WARS: Galactic Racer and Tomb Raider: Legacy of Atlantis.[8] **Positioning:** Arc validation needs separate SR, MFG, and latency telemetry; this reinforces #8 rather than changing its vendor-native-kernel direction.
- **Apple / Metal:** Apple now documents a redesigned MetalFX temporal upscaler that uses Neural Engine plus M5 Pro/Max Neural Accelerators; the framework adds dynamic-resolution subrectangles, motion-vector input, and distortion fields. **Competitive implication:** the Metal port should treat engine MVs and dynamic-resolution regions as first-class inputs, but this is platform-native competition rather than evidence that OSS kernels exist.[19]
- RSS-only FSR 4.1 performance and DLSS 4.5 game-support reports are leads, not inputs to OSS architecture decisions; no primary AMD, Unreal, Unity, or GPU-product announcement in the Aug. 4–18 window cleared the verification threshold.[unverified]

## Action Items / Follow-ups
- [ ] **Critical:** restore/restart `srcnn-v7.0-pico-005` or its metrics publication. Step 36,650 has not advanced for 218h; fix dashboard run-label inconsistency at the same time.[1]
- [ ] Land and benchmark #12 → #13 → #14 before a kernel-fusion/quantization decision.[9]
- [ ] Add an ACA-GS-style capacity-budget ablation: per-region primitive count and feature width, hard bounded by temporal identity and L2-residency measurements.[2]
- [ ] Evaluate Amulet-derived future-visibility cache/refresh scheduling against Gaussian-canvas OSS-FX, including disocclusion and dynamic-lighting cases.[4]
- [ ] Run MeanSR as a teacher-free student-training control against reparameterized HAT distillation; report game-VSR latency separately.[3]
- [ ] Add compatibility telemetry that independently exposes SR, frame generation, and low-latency state for the DLL-shim targets.[7][8]

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.04581
[3] https://arxiv.org/abs/2608.09405
[4] https://arxiv.org/abs/2608.10423
[5] https://archive.org/advancedsearch.php?q=gaussian+splatting+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&fl%5B%5D=mediatype&output=json&rows=10
[6] https://api.github.com/search/repositories?q=gaussian+splatting+super-resolution+pushed%3A%3E%3D2026-08-04&sort=updated&order=desc&per_page=10
[7] https://www.nvidia.com/en-us/geforce/news/no-more-room-in-hell-2-hell-let-loose-vietnam-dlss
[8] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
[9] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[10] https://raw.githubusercontent.com/cashcon57/open-supersampling/main/README.md
[11] https://arxiv.org/abs/2608.13502
[12] https://arxiv.org/abs/2608.08553
[13] https://arxiv.org/abs/2608.13460
[14] https://arxiv.org/abs/2605.11489
[15] https://web.archive.org/cdx/search/cdx?url=arxiv.org%2Fabs%2F2606.29400&output=json&filter=statuscode%3A200&fl=timestamp%2Coriginal%2Cstatuscode&limit=2&from=2026
[16] https://arxiv.org/abs/2412.20720
[17] https://github.com/hy300leosquizz-ctrl/ZeroFG
[18] https://github.com/Adam-AtlasSoftware/RDNU
[19] https://developer.apple.com/metal/whats-new/

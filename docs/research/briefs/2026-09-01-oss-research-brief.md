# OSS Research Brief — 2026-09-01

## Project Status Summary
- **Latest commits:** checkout head is `b44278a test: add local Chrome Playwright dashboard smoke`; `git log -10` exposes only that reachable commit.[unverified]
- **Open issues:** **12**. Immediate implementation/perf gates remain #12 packed low-rank canvas layout, #13 Nsight L2 profiling, #14 VRAM footprint, and #3 tile binning; #4/#7/#9 are the student-distillation path.[2]
- **README snapshot:** OSS is pre-alpha v7 N-D `(x,y,t)` Gaussian-canvas teacher research. The ≤1M shipping student and CUDA/HIP/Metal/Level Zero/Vulkan production backends are still unbuilt.[3]
- **Training dashboard:** live `data.json`/`status.json` report `srcnn-v7.0-pico-005` at **36,650 / 100,000**, loss **0.19268**, with Trainer **offline / stuck 53h**; Watcher, CF Worker, and R2 are healthy. This is unchanged from Aug. 31, so the trainer is stalled; the dashboard’s static shell still presents a live/ETA view before the JSON loads.[1][12]

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation](https://arxiv.org/abs/2608.30184)** (Jiahao Wu, Jie Liang, Die Hu, Jiayu Yang, Kaiqiang Xiong, Xiang Li, Xiaoyun Zheng, Chao Wang, Ronggang Wang; 2026-08-31; arXiv:2608.30184) — Time-conditioned anchors localize spatial/temporal support and temporal windowing activates only anchors relevant to a query time for long dynamic sequences.[4] **Why it matters to OSS:** strongest new v7 design lead: test anchor/window residency as an alternative to globally active Gaussians, but retain exact engine-MV warp and require disocclusion/temporal-ID gates before adopting.
- **[SMG: Semantic Motion Graph for Monocular Dynamic Gaussian Splatting](https://arxiv.org/abs/2608.31023)** (Haozheng Yu, Xinyu Yang, Rundong Luo, Jennifer J. Sun, Bharath Hariharan; 2026-08-31; arXiv:2608.31023) — A low-rank semantic motion graph lets reliable nodes guide uncertain Gaussians under occlusion and complex motion.[5] **Why it matters to OSS:** a teacher-side reliability prior for invalid engine MVs/disocclusions; do not import its learned motion estimator where engine vectors are exact.

### Super-Resolution / Upscaling
- No newly published direct game-SR or ≤1M VSR-student paper cleared the architecture/deployment gate. Existing NanoVSR, CDA-VSR, GS-STVSR, GSASR, and QuantSR+ remain better controls; do not replace actual fused 1080p-LR profiling with paper latency claims.[unverified]

### Frame Extrapolation / Generation
- No new material single-network frame-extrapolation result was verified. Recent 4DGS/world-model papers continue to support persistent-state reuse conceptually, but none demonstrates low-latency engine-MV frame extrapolation; v7 fractional-time rendering remains differentiated.[unverified]

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion paper changes priorities. Measure #12 bytes/Gaussian, #3 bin/sort cost, #13 L2 behavior, and raster latency before selecting quantization or fusion boundaries.[unverified]

## Archive.org Findings
- Internet Archive Advanced Search returned **zero** items for the exact requested phrases `"gaussian splatting" super-resolution`, `"frame generation" deep learning`, and `"real-time upscaling" game` in this run.[6]
- Targeted Wayback CDX checks found a June 7 capture of **3DGS³** (`arXiv:2605.11489`) but none for GS-STVSR (`arXiv:2604.18047`). No useful archived/taken-down primary copy was verified for GS-STVSR, GSASR, GRAPE, or STSS; this is archive coverage failure, not an absence/takedown conclusion.[6]

## GitHub / Open Source
- **[IE-SRGS](https://github.com/ZheXues/IE-SRGS)** is a newly surfaced AAAI 2026 3DGS-SR implementation (created Aug. 24, pushed Aug. 29), but has one star and no declared SPDX license.[7] **Watch only:** inspect its internal/external knowledge fusion for teacher-side SR controls; do not import code or treat it as a game-runtime competitor.
- **[OSSS](https://github.com/Lenovive/OSSS)** is a newly surfaced MIT C++20 window-capture upscaler/frame-interpolator (created/pushed Aug. 25) with zero stars. Its Aug. 25 release added macOS/Linux capture/presentation backends around a portable software interpolator.[8] **Competitive signal, not threat:** it validates cross-platform demand for the surface area, but it is capture/optical-flow interpolation rather than OSS’s engine-input Gaussian-canvas path.
- GitHub REST searches found no material new open-source DLSS/FSR/XeSS analysis tool or production-grade Gaussian-SR competitor in the Aug. 25–Sep. 1 window.[unverified]

## News
- **AMD / UE 5.8 (Aug. 27):** the FSR plugin now bundles FSR Upscaling 4.1.1 and ML Frame Generation 4.0.1, extends ML upscaling to RDNA 3/RX 7000, and says FG 4.0.1 improves motion-vector/camera-parameter handling in corner cases.[9] **Threat:** AMD now presents a single engine plugin with upscaling + FG plus driver-delivered updates; OSS needs explicit UE input-contract, motion-vector, latency, and quality baselines.
- AMD’s plugin guide confirms the shipped stack spans UE 5.3–5.8, has automatic FSR 4 selection on RDNA 3/4, and documents WPO velocity omissions as a blur/ghosting source.[10] **Opportunity:** make OSS Capture/DLL-shim validation include WPO/forced-velocity and screen-percentage cases, not only ideal G-buffers.
- NVIDIA’s current DLSS 4.5 baseline remains Dynamic MFG/6× MFG plus second-generation Transformer SR; no fresh Sept. 1 update was verified.[11]
- **Intel / XeSS 3 (Aug. 10):** Intel says its October *STAR WARS: Galactic Racer* launch partnership includes XeSS 3 Super Resolution, Multi-frame Generation, and Xe Low Latency.[13] **Watch:** this increases the importance of OSS’s vendor-neutral input contract and a latency-aware FG comparison; it is a game-partnership announcement, not an OSS-equivalent architecture disclosure.
- No fresh primary Apple/MetalFX update was verified in this scan.[unverified]

## Action Items / Follow-ups
- [ ] **P0 ops:** recover or explicitly terminate/document the v7 trainer; it is still at step 36,650/loss 0.19268 and the live status reports it offline/stuck 53h.[12]
- [ ] **ATGS ablation:** prototype time-conditioned anchor ownership plus query-time window activation; gate on active Gaussians/tile, bytes/Gaussian, raster time, cross-window temporal-ID survival, and disocclusions.[unverified]
- [ ] **Motion-confidence ablation:** after exact engine-MV warp, test an SMG-inspired low-rank reliability graph only for invalid MVs/newly revealed regions; compare against a simple residual/confidence head.[unverified]
- [ ] **UE integration test matrix:** add WPO/static-object velocity, forced velocity, screen-percentage overrides, camera cuts, and FG swapchain behavior to OSS Capture/DLL-shim acceptance tests.[10]
- [ ] **Competitive watch:** periodically inspect IE-SRGS; keep OSSS as a low-confidence surface-area signal, not an architectural benchmark.[7][8]

## Sources

[1] https://opensupersampling.com
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[3] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[4] https://arxiv.org/abs/2608.30184
[5] https://arxiv.org/abs/2608.31023
[6] https://archive.org/advancedsearch.php
[7] https://api.github.com/repos/ZheXues/IE-SRGS
[8] https://api.github.com/repos/Lenovive/OSSS
[9] https://gpuopen.com/learn/amd-fsr-plugin-updated-for-unreal-engine-58/
[10] https://gpuopen.com/learn/ue-fsr/
[11] https://blogs.nvidia.com/blog/dlss-path-tracing-g-sync-pulsar-ces-2026/
[12] https://opensupersampling.org/status.json
[13] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships

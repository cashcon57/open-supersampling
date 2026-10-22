# OSS Research Brief — 2026-09-04

## Project Status Summary

- **Latest commits:** local history currently exposes `b44278a test: add local Chrome Playwright dashboard smoke` as its newest (and only visible) commit.
- **Open issues:** 11. Highest-leverage implementation queue remains #1 low-rank latent rasterizer, #3 counting-sort tile bin, #4 ≤1M student distillation, #7 FSR-4 teacher signal, #8 native backend ports, #12 packed low-rank canvas, and #13 Nsight L2 profiling.
- **README snapshot:** pre-alpha v7 teacher research. The deployment student and native HIP/Metal/Level Zero/Vulkan kernels remain unbuilt; v7 is the N-D `(x,y,t)` Gaussian canvas path.
- **Training dashboard:** direct `data.json`/`status.json` now report `srcnn-v7.0-pico-005` at **36,650 / 100,000**, total loss **0.19268** (SR Charbonnier **0.02360**, FG Charbonnier **0.07643**), but the trainer health row says **offline / stuck for 125h**; watcher, Worker, and R2 are healthy.[1] This is no step/loss movement from the 2026-09-03 brief while the reported stuck duration advanced from 101h to 125h. Treat the dashboard's “training now” copy as stale: trainer recovery is P0 before architecture conclusions.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **M³ISR: A Multi-Modal Multi-View Benchmark for 3D/4D Gaussian Splatting and Feedforward Compression** — Xinhui Liu, Lei Liu, Zhenghao Chen, Lebin Zhou, Wei Wang, Wei Jiang; 2026-08-23; arXiv:2608.22465. A controlled synthetic 3D/4DGS benchmark with synchronized 1080p views, RGB/depth/semantic/motion annotations, and dedicated streaming/compression tracks.[2] **Why it matters to OSS:** newly material evaluation resource for #12-style canvas compression and streaming; its camera-centric 3D/4D workload is not a proxy for engine-MV 2D temporal SR, so use it only as a supplementary representation-efficiency benchmark.
- **QuADA-GS / Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** — Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto; 2026-06-28; arXiv:2606.29400. It predicts a global-budgeted, locally routed Gaussian densification policy from LR input.[5] **Why it matters:** already tracked, but directly supports a bounded-canvas / loss-adaptive-spawn ablation; benchmark against OSS temporal-ID and disocclusion gates.

### Super-Resolution / Upscaling

- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; arXiv:2607.10495 (ECCV 2026). A structurally reparameterized fully convolutional VSR model removes explicit flow/custom ops for deployment; its 644K model reports 27.2 FPS on Jetson Orin NX 16GB at 25 W.[4] **Why it matters:** already tracked but remains the best concrete ≤1M student baseline; reproduce model/operator counts and TensorRT behavior, not its no-engine-MV premise.
- **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling** — Rong Fu, Chunlei Meng, Yangchen Zeng, Xiaowen Ma, Yongtai Liu, Wangyu Wu, Shuo Yin, Zijian Zhang, Sicheng Li, Yingrui Ji, Chenhao Wang, Simon Fong; 2026-08-09; arXiv:2608.08553. Motion-aware latent-state prediction with sparse attention and a quality-versus-temporal-smoothness control.[15] **Why it matters:** already tracked; its controllable temporal trade-off is a teacher-side loss/evaluation lead, not a viable ≤1M runtime model.

### Frame Extrapolation / Generation

- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423. A sparse tiled layered image-space cache predictively fills future-visible shading and claims 60 Hz shading to 240 Hz extrapolation with localized updates.[3] **Why it matters:** already tracked and the closest systems competitor to OSS-FX's “persistent representation + cheap future render” thesis. Deep-read its disocclusion cache validity, tile lifetime scheduler, and latency methodology; compare with exact-MV Gaussian-canvas rendering rather than adding another learned FG network.
- No newly submitted material paper matched “single network” frame generation in this run; the current architectural priority remains validating v7 at `t_query=0.5` after trainer recovery.

### Inference Optimization

- No new material TensorRT/INT8/kernel-fusion paper in the 7–14 day window. NanoVSR remains the relevant deployment reference because structural reparameterization collapses to standard convolutions at inference.[4]
- The arXiv scan also surfaced recent 4DGS residency/compression work already captured in the bibliography (ATGS, CC-4DGS, Atlas). Do not append duplicate entries.

## Archive.org Findings

- Archive.org advanced search was checked for `gaussian splatting super-resolution`, `frame generation deep learning`, and `real-time upscaling game`. The first returned three GitHub snapshot-style records; the second and third were broad/noisy (228 and 107 hits respectively), with no material archived paper or code lead.[14]
- Exact-name searches for GS-STVSR, GSASR, GRAPE, and STSS returned no relevant new Archive.org item. A Wayback CDX lookup confirmed historical GSASR GitHub captures, but the service became temporarily unavailable during the subsequent paper checks. This is an archive availability limitation, not evidence that copies do not exist.

## GitHub / Open Source

- **OptiScaler** remains the meaningful compatibility reference: 10,276 stars, pushed 2026-09-04, GPL-3.0, and bridges DLSS/XeSS/FSR inputs plus frame-generation substitutions.[6] **Opportunity:** study its title/input compatibility matrix for OSS's planned shim; do not inherit GPL code or position OSS as another wrapper.
- **RTX40MFG-minimal** (0 stars, no asserted license) was newly pushed today and claims an untested ~2,400-line Ada MFG core.[7] **Threat level: low.** Treat as a reverse-engineering/integration lead only; do not rely on claims or use code without a license review.
- **dlss-unlocked** (3 stars, MIT) was also pushed today and claims DLSS upscaling, FG, and neural-rendering feature unlocking in DX12 titles.[8] **Threat level: low.** Track as market evidence for drop-in integration demand; validate title coverage and security/anti-cheat implications before drawing product conclusions.
- No credible new open-source DLSS/FSR analysis tool displaced OptiScaler in this window.

## News

- **Competitive threat:** NVIDIA's primary announcement says DLSS 5 uses a real-time model for lighting/material detail, while its current launch post makes NBA 2K27 the initial DLSS 5 title and lists additional DLSS game releases.[9][10] OSS should preserve explicit albedo, lighting, normals, depth, and motion input contracts, but stay focused on faithful reconstruction—generative material alteration is a distinct product category.
- AMD's Redstone documentation remains the primary architectural benchmark: RDNA4-optimized ML upscaling with analytical fallbacks on older architectures and a four-feature neural-rendering suite.[11] No date-qualified AMD FSR update in the last 14 days was found.
- Intel's developer material retrieved remains XeSS 1.3-era (2024), not fresh XeSS/Arc news.[12] No primary Intel update or Unreal/Unity upscaling release was found in the 7–14 day window.
- Apple announced on 2026-09-01 that macOS 27 will be the final macOS release supporting Rosetta; universal/Apple-silicon-native build coverage is now an integration requirement, though this is not a MetalFX performance update.[13]

## Action Items / Follow-ups

- [ ] **P0:** recover/diagnose `srcnn-v7.0-pico-005`; reconcile the stale “live / ETA” dashboard copy with the **125h** offline trainer state.
- [ ] **Deep read:** Amulet—extract its layered-cache invalidation, disocclusion, scheduler, and latency methodology; design an OSS-FX comparison at equal rendered-frame budgets.
- [ ] **#12 evaluation:** adapt M³ISR's compression/streaming measures where possible, while retaining OSS-specific temporal primitive identity, disocclusion, VRAM, and L2-residency acceptance gates.
- [ ] **#4 student:** compare NanoVSR-style reparameterized standard-conv blocks against RepVGG/ECBSR/RepSR candidates under the ≤1M and <2 ms TensorRT constraint.
- [ ] **Positioning watch:** monitor DLSS 5 as a separate, proprietary generative neural-rendering category; OSS's defensible lane is open, deterministic, engine-buffer-conditioned SR + extrapolation.

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.22465
[3] https://arxiv.org/abs/2608.10423
[4] https://arxiv.org/abs/2607.10495
[5] https://arxiv.org/abs/2606.29400
[6] https://github.com/optiscaler/OptiScaler
[7] https://github.com/danzig666/RTX40MFG-minimal
[8] https://github.com/ShyVortex/dlss-unlocked
[9] https://www.nvidia.com/en-us/geforce/news/star-wars-zero-company-aliens-fireteam-elite-blood-of-dawnwalker-dlss
[10] https://www.nvidia.com/en-us/geforce/news/dlss5-breakthrough-in-visual-fidelity-for-games
[11] https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering
[12] https://game.intel.com/us/stories/intel-xess
[13] https://developer.apple.com/news
[14] https://archive.org/advancedsearch.php?q=gaussian+splatting+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&rows=5&output=json
[15] https://arxiv.org/abs/2608.08553

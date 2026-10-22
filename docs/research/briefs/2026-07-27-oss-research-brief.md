# OSS Research Brief — 2026-07-27

_Collection: local Git/README, GitHub REST, rendered dashboard, arXiv abstract pages/search (Atom API was rate-limited), Archive.org Advanced Search/Wayback CDX, GitHub repository search, primary NVIDIA/Intel pages, and Google News RSS. RSS-only items are labelled as leads._

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls/API/dashboard. The preceding series preserves the v7 Pico checkpoint and fixes WSL2 DataLoader, cross-OS resume, FlashAttention-2 fallback, TF32/prefetch, and fp32 smoothing under bf16.
- **Open issues:** **12**. Direct performance/architecture gates are #14 v6.2 VRAM measurement, #13 Nsight L2 profiling, #12 packed low-rank canvas, #9 reparameterized student backbone, #8 cross-vendor ports, #7 FSR 4 teacher signal, and #4 ≤1M distillation.
- **README snapshot:** active `srcnn-v7.0-pico-005` is a HAT-Tiny teacher with native `(x,y,t)` Gaussians and `V_xt` covariance motion. The ≤1M TensorRT-FP8 student and non-CUDA runtimes do not exist. Published held-out validation remains v5 only: 25.703 PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-27):** headline reports `36,650 / 100,000`, loss `0.19268`, viz step `36,000`, ETA ~1.1d. Health separately reports **Trainer offline; stuck at step 36,650 for 2h**; watcher published 1s ago, CF Worker HTTP 200 in 83ms, R2 HTTP 200, and GPU metrics are unavailable. The step/loss remain unchanged from the prior same-day report; the reduced stall counter is a dashboard health-counter reset/inconsistency, not evidence of recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis** — Jiahao He, Yihua Shao, Zhengkai Zhao, Pan Gao, Fei Ma, Jingcai Guo, Hao Tang, Nicu Sebe, Qi Tian; **2026-07-23** (revised 2026-07-24), [arXiv:2607.21448](https://arxiv.org/abs/2607.21448). Hierarchical anchors, local temporal offsets, canonical/residual appearance, and stop-gradient isolation separate stable structure from deformation. **Why it matters:** direct ablation source for v7 canvas stability, `V_xt` motion, and loss-adaptive density; already in the bibliography.
- **Future Rendering ≠ Future Surface** — Yukun Shi, Minglun Gong; **2026-07-23**, [arXiv:2607.21471](https://arxiv.org/abs/2607.21471). FutureSurf evaluates dynamic reconstruction beyond observed time and finds render quality does not reliably measure future geometry. **Why it matters:** adopt its held-out-future/error-gap discipline for OSS-FX; already tracked.
- No material Gaussian-temporal paper newer than the 2026-07-25 monitor cutoff was verified. The rate-limited Atom scan re-confirmed GS-STVSR, the closest 2D-Gaussian continuous STVSR precedent, but it is already tracked.

### Super-Resolution / Upscaling
- No newly submitted material game/real-time VSR paper passed the relevance gate. **NanoVSR** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11, [arXiv:2607.10495](https://arxiv.org/abs/2607.10495) — remains the most relevant student control: standard-convolution-only structural reparameterization at 644k parameters. **Why it matters:** direct #9/#4 baseline; already tracked.

### Frame Extrapolation / Generation
- No newly submitted single-network extrapolation result cleared the OSS relevance gate. **Future Rendering ≠ Future Surface** above is the actionable evaluation result; it argues against reporting only rendered-frame metrics for OSS-FX.
- Existing **CDA-VSR** ([arXiv:2603.07694](https://arxiv.org/abs/2603.07694), Yuhang Wang et al., 2026-03-08) remains the relevant engineering control: supplied motion vectors plus residual/frame-type gates reduce alignment cost. **Why it matters:** supports exact-engine-MV coarse warping plus residual/disocclusion gates; already tracked.

### Inference Optimization
- **Finding Fast Filters** — Karima Ma, Andrew Adams, Jonathan Ragan-Kelley; **2026-07-22**, [arXiv:2607.20634](https://arxiv.org/abs/2607.20634). Searches multirate/recurrent/sum/cascade FIR programs and lowers candidates to fused vectorized C++. **Why it matters:** a concrete method to search quality/latency Pareto candidates for EWA/Mip-splat filters; already tracked.
- **Rendering 3D Gaussians on a Graph Processor** — Nicholas Fry, Ignacio Alzugaray, Mark Pupilli, Paul H. J. Kelly, Andrew J. Davison; **2026-07-17**, [arXiv:2607.15951](https://arxiv.org/abs/2607.15951). Tile-owned SRAM rendering isolates routing bandwidth, capacity, and density imbalance bottlenecks. **Why it matters:** frames #12 packed-canvas and #13 cache profiling experiments; already tracked.
- No new TensorRT/INT8/kernel-fusion SR result was verified. **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** ([arXiv:2604.20291](https://arxiv.org/abs/2604.20291), Pham Phuong Nam Nguyen et al., 2026-04-22) remains the deployment-aware quantized-student reference, already in the bibliography.

## Archive.org Findings
- Advanced Search: `"gaussian splatting" super-resolution` yielded **3** archived GitHub-list snapshots; `"frame generation" deep learning` yielded **3** items (including DLSS Swapper/FluidFrames builds and an unrelated video-generation record); `"real-time upscaling" game` yielded **0**. None is a verified archived research-paper version.
- Targeted Wayback CDX: GS-STVSR and STSS returned empty capture sets; GSASR and GRAPE timed out. **No takedown, archival copy, or version change was verified**—this is source unavailability/noise, not evidence of absence.

## GitHub / Open Source
- GitHub REST searches since 2026-07-13 found **no credible new Gaussian-SR repo** and **no new DLSS/FSR/XeSS analysis tool**. Frame-generation hits were unrelated creative-video projects or zero-star OptiScaler forks; not competitive evidence.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains the closest G-buffer-guided open game-upscaler lead: GPL-3.0, 15 stars, pushed 2026-07-24. Its GameIR/vKITTI2/Sintel data contract is useful evaluation intelligence, but GPL prevents Apache-2.0 reuse.
- Intel released [XeSS SDK 3.0.2](https://github.com/intel/xess/releases/tag/v3.0.2) on 2026-07-24: XeLL 1.3.2.10, corrected `minimumIntervalUs` units, fixed a non-Intel-GPU marker-related leak, and unwraps Streamline proxies to avoid a later session crash. **Opportunity:** put native-D3D12/proxy behavior in future DLL-shim acceptance tests.

## News
- **NVIDIA — threat / positioning move (primary):** NVIDIA's [DLSS 5 announcement](https://nvidianews.nvidia.com/news/nvidia-dlss-5-delivers-ai-powered-breakthrough-in-visual-fidelity-for-games) says a fall release will take color + motion vectors and add temporally consistent lighting/material appearance, with masks/intensity/color controls and Streamline integration. It is a planned vendor product, not an OSS-comparable benchmark, but expands the incumbent beyond SR/FG toward neural rendering. OSS needs controlled quality/latency evidence and an engine-input-grounded, deterministic artifact/stability story.
- **NVIDIA — current SDK surface (primary):** the [DLSS developer page](https://developer.nvidia.com/rtx/dlss) lists DLSS 4.5 Dynamic MFG and a second-generation Transformer SR model; its July 2026 plugin package covers UE 5.5–5.8 (SL 2.11.1 / NGX 310.6.0). Streamline/D3D12 interoperability is now integration-table stakes.
- **AMD / Apple / engines:** no primary AMD FSR 4, Apple MetalFX, Unreal, or Unity architectural update was verified in the last 14 days. Google News RSS surfaced Proton GE/OptiScaler FSR 4 support and a claim of Radeon multi-frame generation; treat both as unverified leads, not product facts.

## Action Items / Follow-ups
- [ ] **Critical:** inspect/restart the training metrics producer or 3080 Ti host. Step/loss and GPU telemetry did not move; watcher/R2/Worker health only proves publication infrastructure.
- [ ] Deep-read NanoVSR and benchmark a ≤1M reparameterized-CNN control with exact engine-MV/canvas inputs, TensorRT export, and no custom operators against #9's RepVGG/ECBSR/RepSR candidates.
- [ ] Pair #13 profiling with #12 layout selection: measure L2/DRAM/tile density before testing texture-backed/BCn attribute layouts under random access, gradients, and temporal-shimmer gates.
- [ ] Add XeLL/Streamline proxy and native-D3D12 cases to DLL-shim acceptance tests.
- [ ] Add FutureSurf-style held-out-future evaluation: report future/observed error gaps and at least one metric not optimized by the training loss.
- [ ] Competitive watch: DLSS 5 makes a controlled OSS-vs-vendor bake-off and deterministic stability narrative more urgent.

## Bibliography Update
- No append. Every material paper re-confirmed in this run already has one bibliography entry; avoided duplicates.

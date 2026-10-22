# OSS Research Brief — 2026-07-28

_Collection: local Git/README, GitHub REST, rendered dashboard, arXiv Atom API, Archive.org Advanced Search, GitHub repository search, primary NVIDIA/Intel pages, and Google News RSS. RSS-only items are labelled as leads._

## Project Status Summary
- **Latest commits:** `a97a180` / `fcf8084` add a tailnet-gated training-control API and dashboard panel. Preceding commits preserve the v7 Pico checkpoint and improve WSL2 loading, checkpoint portability, FlashAttention fallback, TF32/prefetch, and bf16 raster-filter precision.
- **Open issues:** **12**. Architecture/performance gates include #14 (v6.2 VRAM measurement), #13 (Nsight L2), #12 (packed low-rank canvas), #9 (reparameterized student), #8 (HIP/Metal/Vulkan/SYCL), #7 (FSR 4 teacher signal), #4 (≤1M distillation), #3 (counting-sort bin), and #1 (R=4–8 rasterizer).
- **README snapshot:** `srcnn-v7.0-pico-005` is the active HAT-Tiny research teacher with native `(x,y,t)` Gaussians and `V_xt` covariance motion. No shipping ≤1M TensorRT-FP8 student or non-CUDA backend exists. Published held-out validation remains v5 only: **25.703 dB PSNR / 0.1666 LPIPS** on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-28):** headline says **step 36,650 / 100,000**, loss **0.19268**, viz step **36,000**, ETA ~1.1d. Health independently says **Trainer offline; stuck at 36,650 for 3h**; watcher published 0s ago, CF Worker is HTTP 200 in 113ms, R2 is HTTP 200, and GPU stats are unavailable. The dashboard's live/ETA copy conflicts with its trainer health; step/loss and absent GPU telemetry show no verified progress. The same-day health counter previously displayed 14h at this unchanged step/loss, so the counter has reset/is stale—not evidence of recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **SubSplat: High-Resolution Pixel-aligned 3DGS via Sub-pixel Gaussian Reparameterization** — Jiun Lee, Jaekwang Kim, Sangmin Lee; 2026-07-23, [arXiv:2607.20813](https://arxiv.org/abs/2607.20813). It splits low-resolution, pixel-aligned Gaussians into fine-grained primitives and aggregates multi-view features to restore structural density without quadratic input-resolution cost. **Why it matters to OSS:** direct fresh ablation inspiration for v7 parent-child/loss-adaptive densification and the packed canvas (#12); it is not a temporal or game-runtime method, so use it as a density-allocation control rather than a competitor.
- The targeted Atom queries for Gaussian-temporal/4D material timed out or returned HTTP 429; this is incomplete coverage, not a negative result. The 2026-07-27 brief already captures **GrainGS** (arXiv:2607.21448), whose canonical/residual split is still the stronger direct temporal-stability reference.

### Super-Resolution / Upscaling
- **No newly verified material paper** because the targeted Atom request was rate-limited. The current relevant student control remains **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11, [arXiv:2607.10495](https://arxiv.org/abs/2607.10495). It is a 644k-parameter standard-convolution VSR model using structural reparameterization. **Why it matters to OSS:** concrete ≤1M baseline for #9/#4; already tracked.

### Frame Extrapolation / Generation
- **No newly verified material single-network extrapolation paper.** The frame query was HTTP 429. **Future Rendering ≠ Future Surface: A Benchmark and Dataset for Dynamic Surface Reconstruction Beyond the Observed Window** — Yukun Shi, Minglun Gong; 2026-07-23, [arXiv:2607.21471](https://arxiv.org/abs/2607.21471) — remains the most actionable recent evaluation lead: rendered quality alone does not reliably capture future-surface quality. **Why it matters to OSS:** add a held-out-future/error-gap evaluation for OSS-FX; already tracked.

### Inference Optimization
- **No newly verified TensorRT/INT8/kernel-fusion SR paper.** The optimization query timed out. Keep the two already-tracked, relevant controls: **Finding Fast Filters** — Karima Ma, Andrew Adams, Jonathan Ragan-Kelley; 2026-07-22, [arXiv:2607.20634](https://arxiv.org/abs/2607.20634), which searches/filter-lowers fused multirate programs; and **Rendering 3D Gaussians on a Graph Processor** — Nicholas Fry, Ignacio Alzugaray, Mark Pupilli, Paul H. J. Kelly, Andrew J. Davison; 2026-07-17, [arXiv:2607.15951](https://arxiv.org/abs/2607.15951), which makes tile-local SRAM, bandwidth, and density imbalance explicit. **Why they matter:** candidates for EWA/Mip filter search and for #12/#13 layout/profiling design.

## Archive.org Findings
- Advanced Search returned **3** noisy GitHub-list snapshots for `"gaussian splatting" super-resolution`, **3** items for `"frame generation" deep learning` (DLSS Swapper/FluidFrames builds plus unrelated video generation), and **0** for `"real-time upscaling" game`.
- Targeted acronym searches are not useful archival evidence: GS-STVSR returned 0; GSASR returned one unrelated CIA record; GRAPE and STSS were heavily polluted by unrelated results. No archived/taken-down paper version, GS-STVSR/GSASR/GRAPE/STSS update, or credible replacement was verified.

## GitHub / Open Source
- GitHub REST search since 2026-07-14 produced one Gaussian-SR hit, `hvcl/3DCT_SR_DDNM_NAB-GS` (Apache-2.0, 4 stars), but it is zero-shot CT SR, not a game/runtime lead. Frame-generation hits were creative-video projects or a zero-star OptiScaler fork. No new credible open DLSS/FSR/XeSS analysis tool appeared.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) is still the closest relevant open game-upscaler lead: GPL-3.0, 15 stars, last pushed 2026-07-24. It has FP16/INT8 paths and its latest commit claims an RDNA WMMA INT8 1×1 convolution validation. **Opportunity:** inspect its data contract and INT8 measurement methodology; **constraint:** GPL blocks direct Apache-2.0 reuse.
- Intel released [XeSS SDK 3.0.2](https://github.com/intel/xess/releases/tag/v3.0.2) on 2026-07-24: XeLL 1.3.2.10, corrected `minimumIntervalUs` units, fixed a non-Intel-GPU marker leak, and unwraps Streamline D3D12 proxies to prevent a later-session crash. **Opportunity:** add native-D3D12/proxy, idle-Present, and timing-unit cases to future DLL-shim acceptance tests.

## News
- **NVIDIA (primary; competitive positioning):** [DLSS 4.5](https://developer.nvidia.com/rtx/dlss) now documents Dynamic MFG, a second-generation Transformer SR model, and Dynamic/6× MFG. The July plugin package contains SL 2.11.1 and NGX 310.6.0; it has UE 5.5 package support. This raises integration-table stakes but is not a directly comparable OSS benchmark. OSS needs controlled quality/latency and deterministic temporal-stability evidence.
- **DLSS 5 reports (RSS-only lead):** several July 21 outlets report SIGGRAPH DLSS 5 demonstrations and selectable neural-rendering modes. No primary NVIDIA DLSS 5 announcement was verified in this run; do not make product/competitive claims from these headlines.
- **FSR 4, Arc/XeSS, Apple/Metal, engines:** no primary AMD FSR 4, Apple MetalFX, Unreal, Unity, or consumer-GPU announcement with new OSS-relevant architecture was verified in the last 14 days. RSS found Proton GE/OptiScaler FSR 4 support and Apple M5/frame-generation claims; both are leads only, not decision-grade evidence.

## Action Items / Follow-ups
- [ ] **Critical:** repair/restart the trainer or metrics producer on the 3080 Ti. Watcher/R2/Worker health only proves the publication path; 36,650/0.19268 and GPU telemetry have not advanced.
- [ ] Deep-read SubSplat's splitting/reparameterization rule as a control for v7 parent-child density; measure quality-per-Gaussian and packed-layout bandwidth, not just image fidelity.
- [ ] Deep-read/benchmark NanoVSR as a ≤1M standard-convolution control with exact engine-MV/canvas inputs, TensorRT export, and no custom operators against #9's RepVGG/ECBSR/RepSR candidates.
- [ ] Couple #13 Nsight profiling to #12 layout selection: measure L2/DRAM/tile density before selecting a packed or texture-backed Gaussian layout.
- [ ] Add XeLL/Streamline proxy, native D3D12, idle-Present marker, and `minimumIntervalUs` unit tests to the DLL-shim integration suite.
- [ ] Add FutureSurf-style held-out-future evaluation: report observed-vs-future error gap plus a metric not directly optimized by the training loss.
- [ ] Re-run arXiv as a narrow `id_list`/category-recent pass after the 429/timeout window clears; do not treat this run as full paper coverage.

## Bibliography Update
- Added one newly surfaced, primary-verified material paper: **SubSplat** (arXiv:2607.20813). All other material leads were already tracked.

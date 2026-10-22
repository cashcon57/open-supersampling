# OSS Research Brief — 2026-10-09

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (local log returns one commit, despite requesting 10).
- **Open issues:** 12 returned; notable #14 VRAM footprint, #13 Nsight L2 profile, #12 packed low-rank canvas, #8 cross-vendor ports, #4 ≤1M student distillation, #7 FSR 4 teacher signal.
- **README snapshot:** reports v7 N-D Gaussian canvas with native time axis; consults ongoing teacher/student distinction. README is project-authored, not a live telemetry source.
- **Training dashboard:** web extraction succeeded. Active `srcnn-v7.0-pico-005` at step 36,650/100,000, loss 0.19268, ETA ~1.1 days; however dashboard explicitly reports **Trainer offline / stuck at step 36,650 for 486h**, GPU stats unavailable. Watcher, CF Worker and R2 origin healthy. ETA is stale; investigate trainer/host before relying on progress. Best validated row remains v5-pixel-temporal at 25.70 dB / 0.1666 LPIPS on held-out TartanAir oldtown; industry rows are non-controlled public-source estimates.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **GS-STVSR** (Mingyu Shi et al.; 2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047), previously in bibliography) is the closest Gaussian-temporal precedent: continuous 2D-Gaussian motion, covariance resampling alignment and adaptive offsets; reports near-constant inference at temporal ×2–×8 and >3× speedup at ×32. **Why it matters:** directly informs v7's native-time canvas; test covariance drift/large-motion ablations and compare optical flow to exact engine MVs. No newer, untracked Gaussian-temporal paper surfaced.
- **GSASR** (Du Chen, Liyi Chen, Zhengqiang Zhang, Lei Zhang; 2025-01-12; [arXiv:2501.06838](https://arxiv.org/abs/2501.06838), ICCV 2025; previously tracked) applies CUDA scale-aware 2D-Gaussian rasterization to arbitrary-scale image SR. **Why it matters:** mature spatial SR baseline/code reference, but not a new temporal competitor.

### Super-Resolution / Upscaling
- **Streaming-Aware Diffusion for Real-Time Video Super-Resolution via Cross-Step Attention** (Harris Partaourides, Sotirios Chatzis; 2026-10-08; [arXiv:2610.11746](https://arxiv.org/abs/2610.11746)) — Reuses denoising features across adjacent frames/steps for streaming VSR and reports >40 FPS at 512² after cold start. **Why it matters:** an interesting teacher/perceptual-quality direction, but diffusion complexity and reported resolution make it unlikely to satisfy OSS's ≤2 ms shipping student budget; verify hardware and latency details before considering.

### Frame Extrapolation / Generation
- **No newly surfaced directly applicable paper.** Recent exact-title search surfaced depth-SR work (FoundDSR, arXiv:2609.32323), but it is RGB-guided depth reconstruction rather than game-color SR; excluded from bibliography. Existing material baseline [Amulet (arXiv:2608.10423)](https://arxiv.org/abs/2608.10423) remains relevant for sparse layered-cache extrapolation, but is not new to this scan.

### Inference Optimization
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495)) — Structural reparameterization collapses a compact temporal model to standard convolutions; abstract reports 644k-param variant at 27.2 FPS on Jetson Orin NX. **Why it matters:** concrete student-backbone/deploy-graph comparator for OSS's ≤1M plan and reparameterization issue #9; not new this week, but newly surfaced by this run and already listed in `BIBLIOGRAPHY.md`.

## Archive.org Findings
- Broad Advanced Search: “gaussian splatting” + “super-resolution” returned 3 archived GitHub-list snapshots; “frame generation” + “deep learning” returned 3 noisy records; “real-time upscaling” + “game” returned 0. No verified archived version of GS-STVSR, GSASR, GRAPE, or STSS identified. Inventory is inconclusive, not evidence of absence.

## GitHub / Open Source
- **Opportunity / watch: [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** — public repo describes an experimental vendor-neutral Vulkan frame-generation engine focused on mobile; 5 stars, Apache-2.0, pushed Oct 9. Adjacent to OSS's extrapolation path, but early POC and FG-only per description; monitor motion input contract and measured latency/quality.
- **[DLSS Unlocked](https://github.com/ShyVortex/dlss-unlocked)** — 778 stars, MIT, updated Oct 8; repo claims access to DLSS/MFG/neural-rendering features on compatible RTX/DX12 games. Compatibility tooling, not an open model/direct architecture competitor.
- **[FoundDSR](https://github.com/yanzq95/FoundDSR)** — 1 star, pushed Sep 25; depth-SR research repo, no license reported by GitHub API. Adjacent rather than direct.
- **[SE-FG](https://github.com/PhoenixTheSage/SE-FG)** — open-source Space Engineers frame-generation project; no stars, MIT, updated Oct 1. Narrow integration lead, not a general engine or model competitor.
- Searches: GitHub REST repository search for `gaussian splatting super-resolution`, `frame generation open source`, and `neural upscaler`, updated since Sep 25; broad results were noisy. No verified new open-source general game SR model surfaced.

## News
- **NVIDIA / DLSS:** NVIDIA states DLSS 5 launched in September 2026, first in NBA 2K27 on RTX 50 / GeForce NOW, as an optional 3D-guided neural-rendering appearance stage in addition to SR/MFG/Ray Reconstruction. [Official source](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/). **Threat:** sets a new appearance-quality bar but is platform-limited and distinct from reconstruction; OSS should articulate its fidelity/authorship boundary.
- **AMD / FSR:** GPUOpen search results describe FSR Redstone SDK 2.2/2.3 with ML Upscaling 4.1/4.1.1 and FG 4.0.1, plus UE 5.8 plugin coverage (UE 5.3–5.8; RX 7000/9000 ML path, analytical fallback). [GPUOpen](https://gpuopen.com/). **Threat:** tighter engine integration and fallback story increase competitive pressure on OSS's portability proposition.
- **Intel / XeSS:** XeSS 3 includes SR, FG and multi-frame generation; Intel documentation says cross-vendor SR uses Shader Model 6.4, while MFG requires Intel hardware. [Intel developer docs](https://www.intel.com/content/www/us/en/developer/topic-technology/gamedev/xess.html).
- **Apple / Metal:** MetalFX currently emphasizes temporal upscaling, motion-vector inputs, dynamic-resolution subrectangles, quantized formats and M5 Pro/Max Neural Accelerators. [Apple](https://developer.apple.com/metal/whats-new/). **Opportunity:** ensure Metal planning covers motion-vector and quantized inference paths.
- **Game engines / recent window:** Unreal Engine 5.8 released October 3 and AMD's FSR plugin lists 5.8 support. Recent searches surfaced multi-vendor upscaler support in Gears of War: E-Day but no confirmed new core DLSS/FSR/XeSS generation launch in the last 7–14 days. Vendor pages above are primary references; launch coverage is secondary.


## Action Items / Follow-ups
- [ ] **HIGH:** Restore/inspect trainer and host: dashboard data identifies step 36,650 as 486h stale despite healthy publishing infrastructure; verify process/checkpoint freshness and correct stale ETA.
- [ ] Benchmark NanoVSR's reparameterized standard-convolution graph against the ≤1M OSS student at target game resolutions; compare temporal metrics and TensorRT latency, not only Orin throughput.
- [ ] Track ZeroFG's Vulkan maturity, MV contract and measured benchmarks against OSS's joint canvas SR/FG proposition.
- [ ] Deep-read GS-STVSR covariance-resampling and motion-adaptive-offset ablations; test against exact engine MVs.
- [ ] Track DLSS 5 as a distinct generative-rendering competitor and define OSS's fidelity/authorship boundary.
- [ ] Continue #12 packed-canvas and #13 Nsight L2 work; require measured frame-time and VRAM before adopting candidate optimizations.

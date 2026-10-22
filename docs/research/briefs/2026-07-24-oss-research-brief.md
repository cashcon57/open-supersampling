# OSS Research Brief — 2026-07-24

_Collection: local repository/README, GitHub REST, rendered dashboard, arXiv recent-list/direct abstract pages, Archive.org Advanced Search, Google News RSS, and official NVIDIA/Intel pages. Broad arXiv Atom queries timed out or returned HTTP 429; only primary-verified material is included._

## Project Status Summary
- **Latest commits:** `a97a180` adds tailnet-gated dashboard controls; `fcf8084` adds their API/panel; `0660f55` preserves the v7 step-5K recovery snapshot; `56a9a65` avoids the WSL2 DataLoader deadlock; `d1255c8` enables cross-OS checkpoint resume; `e1ab7fe` adds FlashAttention-2 fallback handling; `e9b4b65` adds TF32/prefetch; `26cd9da` fixes the PSNR proxy.
- **Open issues (10 shown):** #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M student distillation.
- **README snapshot:** v7 remains a research teacher: native `(x,y,t)` Gaussians, `V_xt` covariance motion, loss-adaptive parent/child density, Mip-Splatting, and α=0.5 OSS-FX supervision. The ≤1M TensorRT-FP8 student and non-CUDA runtimes are not built.
- **Training dashboard:** `srcnn-v7.0-pico-005` reports **35,250 / 100,000**, loss **0.26097**, viz **35,077**, hero `live`, ETA ~1.1d. Health independently reports **Trainer offline, stuck at 35,250 for 30h**; watcher healthy (published 1s ago), CF Worker HTTP 200/97ms, R2 HTTP 200, GPU stats unavailable. Step remains unchanged while displayed loss changed; treat dashboard health/metric state as inconsistent, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis** — Jiahao He, Yihua Shao, Zhengkai Zhao, Pan Gao, Fei Ma, Jingcai Guo, Hao Tang, Nicu Sebe, Qi Tian; **2026-07-23**; [arXiv:2607.21448](https://arxiv.org/abs/2607.21448). It combines a hierarchical anchor scaffold with per-Gaussian temporal deformation, blocks deformation gradients into canonical positions, and splits canonical/residual appearance; its abstract reports 435.6 FPS and 4.67 MB on a synthetic setting. **Why it matters to OSS:** direct ablation reference for keeping the v7 persistent canvas stable while permitting local `V_xt`/temporal motion and photometric variation. Evaluate the canonical/residual split and stop-gradient before adding density complexity. This is dynamic NVS, not screen-space SR.
- The 24 July `cs.CV` recent list was screened for Gaussian/splatting, temporal, SR, interpolation, and frame-generation titles; no additional direct OSS-material paper passed the relevance gate.

### Super-Resolution / Upscaling
- No newly primary-verified OSS-material paper. **NanoVSR** (arXiv:2607.10495) remains the highest-value reparameterized-CNN control for #9/#4.

### Frame Extrapolation / Generation
- No newly verified result. **SPEED** (arXiv:2607.15585) remains an offline high-motion/disocclusion perceptual-teacher and benchmark lead, not a <2ms runtime architecture.

### Inference Optimization
- Targeted Atom queries for TensorRT/INT8/fusion timed out or returned HTTP 429. Do not infer no publications. GrainGS's compact scaffold is a useful canvas-memory reference but not an inference-kernel result.

## Archive.org Findings
- Advanced Search: `"gaussian splatting" super-resolution` returned three archived GitHub-star-list snapshots rather than papers; `"frame generation" "deep learning"` returned DLSS Swapper/FluidFrames builds plus an unrelated video-generation registration; `"real-time upscaling" game` returned 0.
- Targeted terms GS-STVSR returned 0, GSASR an unrelated historical record, while broad GRAPE/STSS results were irrelevant. No archived replacement/takedown evidence was found for GS-STVSR, GSASR, GRAPE, or STSS. This noisy inventory is not evidence of absence/removal.

## GitHub / Open Source
- GitHub REST search with `pushed:>=2026-07-10` found no credible new Gaussian-SR or DLSS/FSR/XeSS-analysis repo. `RDNU` remains the only credible open G-buffer-guided neural-upscaler baseline: GPL-3.0, 15 stars, updated 2026-07-24; it is a comparison/data-contract lead, not evidence of game-ready quality.
- **New integration lead:** [NVOFG](https://github.com/SimonVitzethum/NVOFG) is an Apache-2.0 Linux-first Vulkan rendering middleware claiming native DLSS Frame Generation plus modular FG/upscaling/RR backends. It was created/pushed 2026-07-17 and has one star, so it is not a competitor signal yet. **Opportunity:** inspect its Vulkan/Streamline boundaries for the OSS future Linux/Vulkan DLL-shim equivalent; do not adopt without validating its claimed native-FG path and licensing boundaries.
- `frameflow` (GPL-3.0, zero-star AMD-iGPU frame generation) and `bananascaler` (MIT, zero-star WIP video upscaler) are too immature for competitive claims.

## News
- **NVIDIA / DLSS — verified primary update:** NVIDIA's developer page now documents **DLSS 4.5**: Dynamic MFG, second-generation transformer SR, up to five generated frames per rendered frame, DLSS UE plugins for 5.5–5.8 (package dated 2026-07-21), and Streamline 2.11.1 / NGX 310.6.0. **Threat:** NVIDIA is widening the quality/FG feature gap. **Positioning:** keep base-render FPS, displayed FPS, end-to-end latency, SR quality, and generated-frame artifacts distinct; OSS's claim remains vendor-neutral canvas reuse with exact engine MVs, not headline multiplier parity.
- **AMD / FSR:** no new primary AMD page was verified in this run. Previous FSR SDK 2.3 / FSR FG 4.0.1 observations remain the actionable AMD integration baseline. RSS reports of FSR multi-frame generation are leads only.
- **Intel / Arc — verified:** Intel released Windows Arc driver **32.0.101.8864** on **2026-07-20** for Arc B/A-series and Core Ultra with Arc graphics. Treat Arc/handheld as an active portability target rather than assuming TensorRT-only student performance transfers.
- **Apple / Metal and engines:** no primary Apple MetalFX performance/API update or Unreal/Unity upscaling-specific release was verified in the 14-day window.

## Action Items / Follow-ups
- [ ] **CRITICAL:** inspect/recover the 3080 Ti trainer. It remains at 35,250 and the rendered health now says 30h offline; watcher/CDN health only proves publication. Reconcile metrics generation with the loss-without-step change.
- [ ] Run a GrainGS-inspired ablation: stable canonical canvas + residual appearance channel, stop-gradient temporal/deformation path into canonical positions; measure SR/OSS-FX disocclusion, density, and memory impact against current `V_xt` behavior.
- [ ] Before #12 hardens packed layout, measure v7 canvas/rasterizer memory and L2 behavior (#14/#13); compare scaffold/anchor overhead against the 32B/Gaussian target.
- [ ] Audit NVOFG's Vulkan/Streamline integration as a Linux/Vulkan boundary reference only; validate native-FG behavior and license separation before treating it as reusable runtime code.
- [ ] Re-run narrow arXiv `id_list`/fielded searches after throttling clears; do not add bibliography entries from RSS or search snippets.

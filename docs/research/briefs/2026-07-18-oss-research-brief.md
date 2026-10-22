# OSS Research Brief — 2026-07-18

_Collected 2026-07-18. Sources: local git/README, GitHub REST, Archive.org + Wayback APIs, rendered dashboard, AMD/Apple primary pages, OpenFG's repository/release, and Google News RSS (explicitly labelled below). arXiv Atom searches were retried across the requested query families but returned HTTP 503/429/timeouts; the paper list below preserves only previously primary-verified material, not search snippets._

## Project Status Summary
- Latest commits: `a97a180` tailnet-gated dashboard controls; `fcf8084` training-control API/panel; `0660f55` v7-pico-005 step-5000 snapshot/recovery doc; `56a9a65` WSL2 DataLoader deadlock rollback; `d1255c8` cross-OS checkpoint resume shim; `e1ab7fe` FlashAttention-2 path; `e9b4b65` TF32/prefetch throughput work; `26cd9da` corrected live-PSNR proxy; `83b4f07` fp32 raster smoothing under bf16; `baf4b02` apex redirect.
- Open issues (10): #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy v6.2 launch; #5 no-ML shader; #4 ≤1M HAT-Tiny student distillation.
- README snapshot: pre-alpha v7 research teacher is a native-time N-D Gaussian canvas (`V_xt` covariance correlation, parent/child density, Mip-Splatting filters, α=0.5 OSS-FX supervision). The ≤1M shipping student and all non-CUDA backends remain unbuilt.
- Training dashboard: `srcnn-v7.0-pico-005` remains at **35,200 / 100,000**, loss **0.10166**, viz **35,077**. The page still says `live`/updated now, but its health tooltip says **Trainer offline, stuck at step 35,200 for 48h**; watcher is healthy (published 1s ago), CF Worker 200/86ms, R2 origin 200, and GPU stats are unavailable. This is continued non-progress from yesterday's same step/loss—not recovery; the display's live headline conflicts with trainer health.

## New Papers (arXiv)

_Fresh discovery was blocked by arXiv API HTTP 503, HTTP 429, and a timeout across the consolidated Gaussian-temporal, SR/VSR/VFI, inference-optimization, and game-motion-vector searches. The entries below remain the recent, primary-verified material already tracked in this same-day brief; no unverified candidate is added to the bibliography._

### Gaussian Splatting / Temporal
- **[SPIN-4DGS: Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements](https://arxiv.org/abs/2607.12362v1)** (arXiv:2607.12362v1; 2026-07-14; Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun) — predicts attributes from explicit spatiotemporal positions with a lightweight feed-forward network rather than directly modeling temporal displacement. Why it matters to OSS: direct ablation source for fast motion, large displacements, and whether v7's `V_xt` representation needs an attribute-prediction fallback. Already in `BIBLIOGRAPHY.md`.
- **[HyperGS: Fast and Generalizable Gaussian Video Representation](https://arxiv.org/abs/2607.11500v1)** (arXiv:2607.11500v1; 2026-07-13; Fatimah Zohra, Chen Zhao, Shuming Liu, Yahya Al Malallah, Bernard Ghanem) — a feed-forward 2D-Gaussian video representation that replaces per-video optimization with a factorized spatiotemporal transformer and dynamic geometric regularization. Why it matters: relevant precedent for amortized v7 canvas/state prediction and collapse prevention. Already tracked.
- **[Grassmannian Splatting I: Moving rank-2 Spacetime Surfels for Dynamic Scene Rendering](https://arxiv.org/abs/2607.10489v1)** (arXiv:2607.10489v1; 2026-07-11; Aaron Maurice Berman, Shantanu Dave) — uses rank-3 4D covariance so a time slice becomes a rank-2 moving surfel with closed-form motion and standard rasterizer compatibility. Why it matters: a mathematically concrete alternate primitive/conditioning formulation for v7's time-sliced Gaussian rasterizer. Already tracked.

### Super-Resolution / Upscaling
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495v1)** (arXiv:2607.10495v1; 2026-07-11; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies) — fully convolutional VSR uses structural reparameterization to collapse into standard convolutions; its 644k model reports 28.64 dB on REDS4 and 27.2 FPS on Jetson Orin NX 25W. Why it matters: strongest direct student lead for issue #9 and a TensorRT/HIP/Metal-friendly ≤1M architecture. Already tracked.
- **[AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution](https://arxiv.org/abs/2607.00987v1)** (arXiv:2607.00987v1; 2026-07-01; Geunhyuk Youk, Jeonghyeok Do, Dayeon Kim, Jihyong Oh, Munchurl Kim) — decouples scale-agnostic latent denoising from continuous rendering and adds temporally gated recurrence. Why it matters: teacher-only reference for arbitrary-scale and temporal-consistency losses; unsuitable for OSS's shipping latency path without substantial distillation.

### Frame Extrapolation / Generation
- **[MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation](https://arxiv.org/abs/2606.27718v1)** (arXiv:2606.27718v1; 2026-06-26; Jun-Sang Yoo, Seung-Won Jung) — follows flow-guided trajectories with adaptive selective scans to refine nonlinear motion and occlusion. Why it matters: external OSS-FX baseline and a candidate diagnostic for disocclusion/fast-motion error classes. Already tracked.
- **[URVC: A Unified Real-Time Neural Video Coding Model with Temporal, Spatial, and Perceptual Adaptivity](https://arxiv.org/abs/2607.15033v1)** (arXiv:2607.15033v1; 2026-07-16; Xihua Sheng, Chang Wen Chen) — selects among temporal prediction candidates and exposes spatial/perceptual adaptation in one real-time codec. Why it matters: indirect but useful analogy for per-region canvas density/quality budgets; not a direct SR/FG architecture competitor.

### Inference Optimization
- No newly material TensorRT/INT8/kernel-fusion SR paper beyond NanoVSR surfaced in this run. The two strongest immediately preceding 3DGS compression papers—GPU-friendly graphics texture coding (arXiv:2607.14513) and SpeedyGS (arXiv:2607.12656)—were verified as already added yesterday; no duplicate bibliography entry.

## Archive.org Findings
- Archive.org inventory searches returned 3 noisy GitHub-list archives for `"gaussian splatting" AND "super-resolution"`, three mixed software/OSF items for `"frame generation" AND "deep learning"`, and zero useful results for `"real-time upscaling" AND game`.
- Targeted Wayback confirms archived arXiv snapshots for the known GS-STVSR candidate `2407.17139` (July and December 2024). GSASR's queried GitHub URL had no CDX rows; one other known-paper CDX query timed out. No evidence of a removed or materially revised GS-STVSR/GSASR/GRAPE/STSS primary source.

## GitHub / Open Source
- GitHub REST searches since 2026-07-04 found no new meaningful Gaussian-SR repository or DLSS/FSR/XeSS analysis tool. `neural upscaler` results were six zero-star toy/GAN/WIP projects; `frame generation open source` returned one unrelated pentest repo.
- **OpenFGG/openfg-app** is nevertheless a concrete ecosystem signal: its July 18 **v0.9** release adds emulator support and HDR mode to its Android OpenGL ES/Vulkan interpolated-FG module (18 stars, Android 10+, root/Zygisk). It is not an OSS architecture threat, but demonstrates demand for user-space/interception-style FG controls and test telemetry. Continue treating OptiScaler-style interception as the desktop DLL-shim compatibility precedent.
- Official repository activity is not a product release signal: `NVIDIA/DLSS` last pushed June 23, `FidelityFX-SDK` June 24, and `intel/xess` April 17.

## News
- **AMD — verified primary update:** [FSR SDK 2.3](https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/) brings ML-powered **FSR Upscaling 4.1.1 to RX 7000/RDNA 3**, plus Frame Generation 4.0.1 and Ray Regeneration 1.2 improvements for RDNA 4. This broadens AMD's ML-SR hardware reach and materially raises OSS's cross-vendor bar; benchmark student quality/latency on RDNA 3, not only RDNA 4.
- **AMD kernel opportunity:** GPUOpen's current RDNA4 WMMA guidance documents FP8/INT8 fused-GEMM memory-layout techniques, and MiniDXNN offers native HLSL/DX12 MLP inference on RX 9000 matrix cores. Evaluate these as a future HIP/HLSL student backend instead of assuming a CUDA/TensorRT-shaped port.
- **Apple — verified primary platform posture:** Apple continues to document shader-side ML inference and MetalFX upscaling, plus Game Porting Toolkit 4 command-line Metal tools. This supports planning a native Metal student/raster path, but no dated MetalFX upscaler release was verified in this window. The targeted NVIDIA and Intel URLs now return 404, so no fresh primary DLSS/XeSS release claim is made.
- **NVIDIA/FSR MFG headlines — RSS-only leads:** Google News carried Gears of War: E-Day DLSS 4.5 support and AMD 8x-MFG claims, but the NVIDIA target page was 404 and no primary AMD 8x confirmation was found. Do not use either as product facts yet.
- No primary Unreal or Unity upscaling-specific update was found in the 14-day window.

## Action Items / Follow-ups
- [ ] **Critical:** recover/inspect the 3080 Ti trainer. The dashboard health counter advanced from 24h to 48h while step/loss/viz did not move; watcher/CDN health does not mean training health.
- [ ] Deep-read SPIN-4DGS and Grassmannian Splatting for a v7 fast-motion/time-conditioning ablation; compare memory and standard-rasterizer compatibility against full Cholesky `V_xt`.
- [ ] Reproduce NanoVSR's reparameterized standard-conv blocks as the first issue #9 student baseline, preserving OSS motion/G-buffer inputs where beneficial.
- [ ] Add an AMD RDNA3 measurement target: FSR 4.1.1 now covers it, so vendor-neutral positioning needs a concrete ROCm/HIP or HLSL plan.
- [ ] Watch FSR 8x MFG and DLSS 4.5 claims only until primary vendor release notes are available.

## Bibliography Update
- No append: every paper above that is materially relevant and suitable for the bibliography was already present; this run verified title-level deduplication.

# OSS Research Brief — 2026-07-20

_Collection: local repository/README, GitHub REST, rendered dashboard, arXiv Atom, Archive.org Advanced Search + CDX, official NVIDIA/AMD pages, and Google News RSS. RSS-only reporting is labelled as such._

## Project Status Summary
- **Latest commits:** `a97a180` tailnet-gated dashboard controls; `fcf8084` training-control API/panel; `0660f55` preserved v7 step-5K recovery snapshot; `56a9a65` WSL2 DataLoader rollback; `d1255c8` cross-OS checkpoint resume; `e1ab7fe` FlashAttention-2 path; `e9b4b65` TF32/prefetch; `26cd9da` corrected live PSNR proxy; `83b4b02` apex redirect.
- **Open issues (12):** #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M student distillation; #3 counting-sort tile bin; #1 R=4–8 low-rank rasterizer.
- **README snapshot:** v7 is a research/teacher model with native `(x,y,t)` Gaussian primitives, `V_xt` covariance motion, loss-adaptive parent/child density, Mip-Splatting and α=0.5 OSS-FX supervision. The ≤1M shipping student and all non-CUDA paths remain unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains **35,200 / 100,000**, loss **0.10166**, viz **35,077**. The headline still says `live`/“updated just now,” but health says **Trainer offline, stuck 96h at step 35,200**; watcher is healthy (published 1s ago), CF Worker is 200/89ms, and R2 is 200. GPU stats are unavailable. This is continued outage, not a recovery.
- **Refresh note:** targeted arXiv Atom queries and the candidate-ID batch are rate-limited/timeout-prone today (429/timeouts), so no search-snippet-only paper was promoted; the primary-metadata entries below are retained from the earlier verified same-day capture.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting](https://arxiv.org/abs/2607.15542v1)** — **Damani Mguni-Coker**, 2026-07-17, arXiv:2607.15542. Spatially truncated variational inference and reassignment reduce reported continual 3DGS reconstruction from ~84s to ~50ms/frame on an RTX 3070 Ti. **Why it matters to OSS:** a concrete candidate for canvas-history truncation/reassignment and bounded-memory updates; deep-read the locality criterion before issue #12’s packed-layout work.
- **[Grassmannian Splatting I: Moving rank-2 Spacetime Surfels for Dynamic Scene Rendering](https://arxiv.org/abs/2607.10489v1)** — **Aaron Maurice Berman, Shantanu Dave**, 2026-07-11, arXiv:2607.10489. Closed-form rank-3 spacetime covariance yields rank-2 moving surfels per time slice without a learned deformation field. **Why it matters to OSS:** already in the bibliography, but it is the closest mathematical contrast to v7’s full-rank native-time `V_xt` primitive; compare slice conditioning, static/moving continuity and its Schur-denominator regularizer.
- Other fresh Gaussian-query hits were robotics, radar, meshing, or generic reconstruction and did not clear the direct OSS transfer bar. `Instant NuRec`, `HyperGS`, SPIN-4DGS and related recent material are already tracked.

### Super-Resolution / Upscaling
- No newly submitted material paper cleared the relevance gate. **NanoVSR** remains the important already-tracked ≤1M/standard-convolution student reference.

### Frame Extrapolation / Generation
- **[SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation](https://arxiv.org/abs/2607.15585v1)** — **Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu**, 2026-07-17, arXiv:2607.15585. One-step pixel-space diffusion VFI uses progressive patch scaling and noise-update-only attention; the abstract claims 8.8% lower LPIPS and 63.3% faster inference on SNU-FILM. **Why it matters to OSS:** not a shipping path at <2ms, but a potential offline/perceptual teacher or high-motion/disocclusion quality reference for OSS-FX.
- No new real-time, game-input-aware frame-extrapolation competitor was found. **MASS** and **GFFE** remain the direct external artifact/occlusion baselines.

### Inference Optimization
- No fresh TensorRT/INT8/kernel-fusion paper specific enough to add. The deployment implication is unchanged: use NanoVSR-style structural reparameterization as the student baseline, then measure real TensorRT/kernel latency rather than infer it from parameter count.

## Archive.org Findings
- Advanced Search inventory: `"gaussian splatting" super-resolution` returned three noisy archived GitHub-star lists; `"frame generation" deep learning` returned DLSS Swapper/FluidFrames binaries plus an unrelated OSF diffusion record; `"real-time upscaling" game` returned zero.
- Targeted CDX refresh: GSASR’s GitHub URL has five retained 200 captures (2025-03-13 through 2026-06-07); GRAPE and generic STSS queries returned no captures, while GS-STVSR timed out. This is not evidence of takedown; no archived primary paper/version newly materialized.

## GitHub / Open Source
- GitHub REST results available before rate limit: the Gaussian-SR query returned none, and the sole `frame generation open source` hit was an irrelevant zero-star pentest project. The `neural upscaler` and DLSS/FSR/XeSS-analysis queries then hit GitHub API 403 rate limits, so no absence claim is made for those two categories; no credible new competitor was verified.
- Existing open-source watchlist remains RDNU and OpenFG; neither was superseded by a verified new repository in this pass.

## News
- **Verified — NVIDIA:** the [official DLSS page](https://developer.nvidia.com/rtx/dlss) now documents DLSS 4.5’s Dynamic MFG, second-generation transformer SR, and 6× MFG; UE 5.5–5.7 plugin package **v8.7.0** is dated July 2026. **Threat:** OSS’s unified SR+FX positioning must compete against an actively integrated, more capable vendor two-path stack; benchmark base-frame latency and artifact behavior, not displayed FPS.
- **Verified — AMD:** [FSR SDK 2.3](https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/) (2026-06-24) brings ML FSR Upscaling 4.1.1 to RX 7000/RDNA3, updates FSR Frame Generation to 4.0.1 with motion-vector/camera input improvements, and updates Ray Regeneration 1.2. **Opportunity/threat:** move RDNA3 from future support to a first HIP/HLSL benchmark target; its MV/camera input contract is a direct OSS integration comparison point.
- **RSS-only:** reports of AMD 8× MFG and Linux DLSS Updater/Proton panel changes appeared in the 7–14 day feed, but no primary AMD or project release was verified. Treat as leads only.
- No primary Intel XeSS, Apple MetalFX, Unreal, or Unity upscaling-specific update was identified in the window.

## Action Items / Follow-ups
- [ ] **Critical:** inspect/recover the 3080 Ti trainer. The current visible counter is **96h** with unchanged step/loss/viz; watcher/CDN health is unrelated to training health.
- [ ] Deep-read ImprovedVBGS’s spatial truncation/reassignment; define a v7 ablation that measures bounded active-canvas size, quality, and temporal artifact impact before committing issue #12’s packed layout.
- [ ] Implement #9’s reparameterized student baseline around standard TensorRT-friendly convolutions, then compare against HAT teacher quality using exact MV/G-buffer input subsets.
- [ ] Add RDNA3 to the cross-vendor benchmark matrix and scope #8 as a HIP/HLSL feasibility spike; AMD’s FSR 4.1.1 coverage raises the immediate bar.
- [ ] Add Grassmannian slice-conditioning and SPEED’s high-motion/disocclusion cases to the OSS-FX paper/benchmark matrix; do not adopt diffusion in the shipping path.

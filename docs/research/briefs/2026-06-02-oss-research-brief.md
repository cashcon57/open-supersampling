# OSS Research Brief — 2026-06-02

## Project Status Summary
- **Latest commits (last 10):**
  - `0660f55` — wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` — trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` — trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` — hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` — trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  - `26cd9da` — dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` — v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` — worker: apex .com → .org 301 redirect
  - `eb0ec2e` — trainer: bf16 autocast around forward + loss
  - `5c2f2e7` — trainer: add throughput baselines (cudnn.benchmark, TF32, torch.compile)

- **Open issues (10 total):** #14 (v6.2-pico-002 VRAM footprint), #13 (Nsight L2 hit-rate profile), #12 (packed low-rank canvas layout), #10 (decoupled G-buffer encoder ablation), #9 (RepVGG/ECBSR/RepSR ablation), #8 (cross-vendor kernel ports), #7 (FSR 4 distillation pipeline), #6 (v6.2-pico-002 launch), #5 (No-ML shader variant), #4 (HAT-Tiny student distillation ≤1M params)

- **Training dashboard:** v7.0-pico-005 stalled at step 9,100 (**stale 66h**, trainer degraded). Current loss 0.12692. Target 100K steps, 9.1% complete. PSNR 25.70 dB on held-out TartanAir oldtown. Cost so far: $0.23 / 3.79 GPU-hours (3080 Ti). **Note: trainer has been stuck for nearly 3 days — likely needs manual restart.**

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[GS-STVSR]** Shi et al. (arXiv:2604.18047, Apr 2026). Already in BIBLIOGRAPHY. **No changes.** First 2DGS-based C-STVSR framework — covariance resampling alignment + optical flow motion module. Closest academic work to OSS's Gaussian-canvas approach. Still the paper to watch for covariance-handling techniques.

- **[RetimeGS]** Wang et al. (arXiv:2603.13783). **ACCEPTED CVPR 2026.** Continuous-time 4DGS with temporal anti-aliasing via triple-rendering supervision. Status unchanged from yesterday.

- **[GaussianZoom]** (arXiv:2605.18252, May 2026). Progressive zoom-in generative 3DGS reconstruction with geometry-guided iterative refinement. Why it matters: Progressive refinement strategy could inform OSS's multi-scale training curriculum — start at low res and "zoom in" rather than random-crop at full HR. But this is generative (diffusion-based), not real-time.

- **[Voronoi-guided Bilateral 2DGS for Hyperspectral SR]** Zhang et al. (arXiv:2604.17727, Apr 2026). GaussianHSI — Voronoi-guided selection + reference-aware bilateral weighting for arbitrary-scale hyperspectral SR. Why it matters to OSS: Voronoi-guided Gaussian-to-pixel association is a novel idea that could replace OSS's tile-based rasterization if applied to RGB. Bilateral weighting (geometric relevance × LR feature consistency) handles the spectral-fidelity problem — analogous to OSS's need to preserve G-buffer feature consistency across upscaled frames. Niche application (hyperspectral) but the spatial reconstruction primitive is transferable.

- **[SuperGaussian]** (arXiv:2406.00609v3, updated 2026). Repurposing video diffusion models for 3D super-resolution. Marginal relevance — diffusion-based, not real-time.

### Super-Resolution / Upscaling

- **[ContinuousSR (Pixel-to-Gaussian)]** Peng et al. (ICLR 2026 Poster, arXiv:2503.06617). Already in BIBLIOGRAPHY. Single-pass 2D Gaussian field construction, **1ms per scale** arbitrary-scale rendering, 19.5× speedup. No updates. Remains the closest work to OSS-Pico rasterization tier.

- **[NTIRE 2026 Efficient SR Challenge Report]** Ren et al. (arXiv:2604.03198, CVPRW 2026). Already in BIBLIOGRAPHY. 95 registrants, target 26.90 dB PSNR. **Key finding:** winning solution SPANV2 from XiaomiMM used a custom fused CUDA kernel that eliminates redundant DRAM round-trips. Why it matters: Fused-kernel design pattern directly applicable to OSS student model inference.

- **[EfRLFN]** Bogatyrev et al. (arXiv:2602.11339v2, Feb 2026). Already in BIBLIOGRAPHY. 15% faster than RLFN via Efficient Channel Attention + tanh activations. No updates.

- **[NTIRE 2026 Mobile Real-World Image SR Challenge]** Li et al. (arXiv:2604.17306, CVPRW 2026). ×4 mobile SR with unknown degradations. Pico-tier relevance. No updates.

- **[NTIRE 2026 3D Content Super-Resolution Challenge]** CVPRW 2026. New this cycle — first NTIRE challenge specifically for 3D content SR. **Winner: WaveletSR** (Wang et al.) for blind SR on noisy 3DGS. Why it matters: Validates 3DGS super-resolution as a growing subfield with dedicated benchmarks. OSS's 2D Gaussian canvas SR is adjacent and benefits from the same ecosystem growth.

- **[INT8-SR]** Nguyen et al. (arXiv:2604.20291, MAI Workshop @ CVPR 2026). Already in BIBLIOGRAPHY. Deployment-aware INT8 quantization with Mamba-based teacher distillation. 30.006 dB / 0.857 SSIM in fixed-shape INT8. **Key insight for OSS:** Three-stage pipeline (spatial→fidelity→QAT) with teacher guidance through all stages. OSS should consider continuing HAT-Tiny teacher supervision during the student's quantization-aware training phase, not just during the initial distillation.

### Frame Extrapolation / Generation

- **[Real-time Video Prediction via Fast Video Interpolation]** (arXiv:2503.23185v2, updated 2026). Extends frame interpolation models to video prediction (extrapolation) via three training methods — conv-only FI model repurposed for next-frame prediction. Why it matters: Demonstrates that interpolation models can be repurposed for extrapolation without architecture changes — supports OSS's thesis that SR+FG can share one primitive. But quality likely degrades beyond 1-2 frames.

- **[Accelerating Autoregressive Video Generation via Selective Computation]** (arXiv:2604.02979, Apr 2026). Not all frames deserve full computation — selective computation + predictive extrapolation for autoregressive video generation. Why it matters: The "selective computation" concept mirrors OSS's idea that extrapolated frames need less compute (α<1 renders cheaper). Could inform adaptive quality scaling.

- **[Playing with Transformer at 30+ FPS via Next-Frame Diffusion]** (arXiv:2506.01380). 310M transformer achieving 30+ FPS autoregressive video generation on A100. Not real-time for OSS's target (<2ms on consumer GPUs) but shows transformer-only generation approaching interactive rates.

- **[Envisioning the Future, One Step at a Time]** (arXiv:2604.09527, Apr 2026). Long-range video prediction method. Academic, not real-time, but the long-range aspect is interesting for OSS-FX multi-frame extrapolation.

- **[X4Rate]** Dixit & Sarangi (VISAPP 2026). Scene-adaptive extrapolation for 4× frame rate. Already in BIBLIOGRAPHY.

### Inference Optimization

- **[NTIRE 2026 Winner — Fused CUDA Kernel]** SPANV2 from XiaomiMM eliminated DRAM round-trips via custom fused CUDA kernel. **Not a paper per se but a key technique pattern.** Why it matters: OSS's student model inference (rasterization + backbone) will be I/O bound on the GPU memory bus. Kernel fusion — combining rasterization reads with first conv layer — could reduce the gap between OSS and DLSS/FSR latency.

- **[FlashFormer: Whole-Model Kernels for Low-Batch Inference]** (arXiv:2505.22758, May 2026). Strategy for writing a single fused kernel that runs the entire forward pass of a transformer. Why it matters: If OSS ever needs to run a mini-transformer (attention-based student component) at inference, whole-model kernel fusion is the path to <1ms.

- **[CUTLASS v4.5.2]** NVIDIA (May 2026). Blockscaled FP8 GEMM for Blackwell. Already in BIBLIOGRAPHY. No major update.

## Archive.org Findings

No new papers discovered on archive.org. Searches for "gaussian splatting" super-resolution, "frame generation" deep learning, and "real-time upscaling" game returned only noise (old textbook scans, unrelated archives). GS-STVSR, GSASR, ContinuousSR, and WaveletSR are all accessible via arXiv directly — no takedowns to recover.

## GitHub / Open Source

- **[gsplat v1.5+]** nerfstudio-project/gsplat (Mar 2026): Added LiDAR rasterization for 3DGS — spinning-lidar camera models, eval3d rendering, depth/hit-distance modes. Not directly relevant to OSS (LiDAR-specific) but shows gsplat ecosystem is actively expanding. The 2DGS rasterizer (which OSS depends on) is stable.

- **[gsplat-2dgs fork]** tongji-rkr/gsplat-2dgs: CUDA accelerated 2D Gaussian rasterization fork. Actively maintained as of 2026. OSS should monitor this fork — if their 2D rasterizer gets performance improvements, OSS benefits directly since the v7 architecture uses 2D-like Gaussian primitives.

- **[fast-gaussian-rasterization]** dendenxu/fast-gaussian-rasterization: Geometry-shader-based global CUDA sorted 3DGS rasterizer. 5-10x speedup over the reference. If the 2DGS fork adopts similar techniques, it could unlock significant rasterization speed gains for OSS.

- **[DLSS Swapper]** Now supports FSR, XeSS, and DLSS updating. Validates the DLL-shim ecosystem — OSS's integration path (DLL replacement) is proven viable by existing tools.

- **[OptiScaler FSR4 Compatibility List]** Growing list of games where FSR4 injection works. Validates injection-based upscaler deployment. OSS's planned DLL-shim runtime follows the exact same pattern.

## News

### DLSS 4.5 / NVIDIA (CRITICAL — Competitor Move)
- **DLSS 4.5 Dynamic Multi Frame Generation** went live (late May 2026) — now supports **5× and 6× multipliers**, dynamic frame pacing matched to monitor refresh rate. Requires RTX 50-series.
- **DLSS 4.5 Ray Reconstruction** unveiled at COMPUTEX 2026 (last week) — **2nd gen transformer model**, broader training data, improved ray-traced/patht-raced denoising. Launching August 2026.
- NVIDIA explicitly positioned DLSS 4.5 as a stepping stone toward **DLSS 5** (Fall 2026, real-time neural rendering — "GPT moment for graphics").
- **RTX Spark** announced at COMPUTEX — AI-powered creative tool, not directly upscaling-related but shows NVIDIA's AI-first GPU direction.
- **Threat assessment:** DLSS 4.5's 6× MFG + 2nd gen transformer sets an aggressive moving target. If DLSS 5 lands with real-time neural rendering, the quality gap between DLSS 4.5 (transformer SR + MFG) and OSS v7 (Gaussian canvas, PSNR 25.70) widens further. However, DLSS is NVIDIA-exclusive — OSS's cross-vendor value proposition stands.

### FSR 4 / AMD
- **FSR 4 (Redstone)** now in 300+ games (up from 30 at launch). ML-based upscaling competing with DLSS 3/4 quality.
- AMD announced plans to **open FSR 4 to competitor GPUs** (including NVIDIA). If materialized, this closes the "vendor-neutral" gap that OSS markets against — but also validates the demand for cross-vendor solutions.
- **FSR 4.1** officially extended to RX 7000 (RDNA 3) and RX 6000 (RDNA 2) series — June/July 2026.
- **FSR 4.1 FP8** support — uses the FP8 hardware found in RDNA 4, Ada Lovelace, and Blackwell GPUs.
- **Threat assessment:** FSR 4 reaching 300 games + cross-vendor openness makes it a formidable competitor in the "vendor-neutral upscaler" space. OSS's differentiation becomes: (1) Apache 2.0 license vs AMD's proprietary ML model, (2) unified SR+FG architecture vs FSR's separate SR + FG pipelines, (3) Gaussian-based anti-aliasing.

### XeSS 3 / Intel
- **XeSS 3 Multi-Frame Generation** now available via WHQL drivers (32.0.x) — supports Arc A-series, B-series, and Core Ultra iGPUs.
- XeSS 3 MFG can be injected into **any title with XeSS 2 support** without per-game patches.
- Intel Arc B580 confirmed as the budget GPU target — sub-$250 card running MFG.
- **Threat assessment:** Intel entering the MFG space with open GPU support (XeSS runs on all vendors) creates a three-way battle. XeSS 3's injection approach (works on any XeSS 2 title) mirrors OSS's planned DLL-shim deployment — Intel is proving the market for injection-based upscaler deployment.

### Apple / MetalFX
- **Metal 4** (WWDC 2025) added MetalFX Frame Interpolation and denoising. **WWDC 2026 is next week (June 9)** — expect MetalFX updates, possibly hardware-accelerated neural upscaling for M4/M5.
- Metal 4's "shader ML" allows embedding neural networks directly within shaders — Apple's answer to NVIDIA's Tensor Cores.

### GPU Hardware
- No major GPU launches this week. RTX 5060/5050 rumored for late June. AMD RDNA 4 mid-range (RX 9600) also rumored.
- NVIDIA RTX 50-series supply remains constrained.

## Action Items / Follow-ups

- [ ] **CRITICAL: Restart v7-pico-005 training.** Trainer stuck at step 9,100 for 66h. Check WSL2 host, restart the training process.
- [ ] **Deep-read: WaveletSR (NTIRE 2026 3D Content SR winner).** Blind SR on noisy 3DGS could inform OSS's G-buffer denoising approach for OSS-RR (ray reconstruction).
- [ ] **Evaluate: Fused kernel pattern from SPANV2.** The NTIRE 2026 winner used custom fused CUDA kernels to eliminate DRAM round-trips — this is directly applicable to OSS student model inference.
- [ ] **Monitor: DLSS 4.5 Ray Reconstruction (Aug 2026).** 2nd gen transformer denoiser — if it ships with source-available SDK components, reverse-engineering the denoising architecture could inform OSS-RR design.
- [ ] **Track: FSR 4 cross-vendor openness.** If AMD ships FSR 4 on NVIDIA GPUs (FP8), it partially validates OSS's cross-vendor thesis but also weakens OSS's differentiation. Prepare messaging: OSS is Apache 2.0 (no SDK contract) vs AMD's proprietary ML model + API contract.
- [ ] **Investigate: Voronoi-guided Gaussian selection (GaussianHSI).** The Voronoi-based pixel-to-Gaussian association is a novel alternative to OSS's tile-based rasterization. Could reduce overdraw and improve cache locality.
- [ ] **Prepare: WWDC 2026 (June 9).** Watch for MetalFX updates — if Apple ships neural upscaling hardware acceleration, OSS needs a Metal kernel sooner.
- [ ] **Archive: GaussianHSI paper (2604.17727)** — add to BIBLIOGRAPHY under Gaussian Splatting + Temporal if it passes deep-read.

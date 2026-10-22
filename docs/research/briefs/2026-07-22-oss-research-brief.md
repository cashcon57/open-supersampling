# OSS Research Brief — 2026-07-22

_Collection: local repository/README, GitHub REST, rendered dashboard, arXiv Atom plus primary abstract pages, Archive.org Advanced Search, official NVIDIA/AMD/Intel pages, and Google News RSS. RSS-only items are labelled as leads._

## Project Status Summary
- **Latest commits:** `a97a180` adds tailnet-gated training controls; `fcf8084` adds the control API/panel; `0660f55` preserves the v7 step-5K recovery snapshot; `56a9a65` reverts the WSL2-deadlocking DataLoader; `d1255c8` adds cross-OS checkpoint-resume support; `e1ab7fe` adds a FlashAttention-2 path; `e9b4b65` enables TF32/prefetch; `26cd9da` corrects the PSNR proxy.
- **Open issues (10 shown):** #14 v6.2 VRAM measurement; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer ablation; #9 reparameterized student; #8 HIP/Metal/Vulkan/SYCL ports; #7 FSR4-teacher distillation; #6 legacy launch; #5 no-ML shader; #4 ≤1M student distillation.
- **README snapshot:** v7 remains the research teacher: native `(x,y,t)` Gaussians, `V_xt` covariance motion, loss-adaptive parent/child density, Mip-Splatting, and α=0.5 OSS-FX supervision. No shipping student or non-CUDA runtime exists yet.
- **Training dashboard:** active run remains `srcnn-v7.0-pico-005`, **35,200 / 100,000**, loss **0.10166**, viz **35,077**. The hero says `live`/ETA ~1.1d, but health says **Trainer offline; stuck for 144h**. Watcher is healthy (published 1s ago), CF Worker is HTTP 200/88ms, R2 origin is HTTP 200, and GPU stats are unavailable. The unchanged step/loss/viz and increased counter indicate continuing trainer outage, not recovery.

## New Papers (arXiv)

_No new submission after the 2026-07-21 brief passed the direct OSS relevance gate. The following recent, primary-verified papers remain the material reading queue; all are already in `BIBLIOGRAPHY.md`, so no duplicate bibliography entry was added._

### Gaussian Splatting / Temporal
- **[SPARE-GS: Structural Parsimony and Resource Efficiency for 3D Gaussian Splatting](https://arxiv.org/abs/2607.16624)** (Zhang Chen, Shuai Wan, Fuzheng Yang, Jiazhi Xia, Weiyao Lin, Junhui Hou; 2026-07-18; arXiv:2607.16624) — Global budget-constrained regional quotas coordinate densification, pruning, and termination; its abstract reports 30.38% fewer Gaussians and 23.81% less training time while improving PSNR. **Why it matters to OSS:** highest-priority ablation against v7 loss-adaptive parent/child density and issue #12’s fixed packed-canvas budget.
- **[Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements (SPIN-4DGS)](https://arxiv.org/abs/2607.12362)** (Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun; 2026-07-14; arXiv:2607.12362) — Learns attributes from explicit spatiotemporal positions through a lightweight network rather than temporal displacement modeling, reporting a +1.83 PSNR advantage over D3DGS on Basketball. **Why it matters to OSS:** direct stress-test reference for fast motion, `V_xt`, and canvas attributes at disocclusions; investigate a lightweight residual attribute head, not a replacement for exact engine-MV warping.
- **[Grassmannian Splatting I: Moving rank-2 Spacetime Surfels for Dynamic Scene Rendering](https://arxiv.org/abs/2607.10489)** (Aaron Maurice Berman, Shantanu Dave; 2026-07-11; arXiv:2607.10489) — Parameterizes motion as rank-3 spacetime covariance so each time slice is a rank-2 surfel, with closed-form conditioning and standard rasterizer compatibility. **Why it matters to OSS:** a mathematically concrete alternative to full-rank time-sliced primitives; useful only as a covariance/raster ablation, since OSS is screen-space 2D rather than novel-view 3DGS.

### Super-Resolution / Upscaling
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; arXiv:2607.10495) — A fully convolutional VSR model uses structural reparameterization to collapse to ordinary convolutions at inference; the 644K model reports 28.64 dB and 27.2 FPS on Jetson Orin NX at 25W. **Why it matters to OSS:** best concrete control for #9/#4: reproduce its reparameterized CNN under the actual RGB/depth/MV G-buffer contract before inventing a custom ≤1M student.

### Frame Extrapolation / Generation
- **[SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation](https://arxiv.org/abs/2607.15585)** (Zihao Zhang, Haoyu Zhao, Siqian Yang, Yidi Wu, Yudong Jiang, Zuxuan Wu; 2026-07-17; arXiv:2607.15585) — One-step pixel diffusion combines dynamic patch scaling with noise-update-only attention and reports better perceptual quality/faster inference than its VFI comparators. **Why it matters to OSS:** an offline high-motion/disocclusion teacher and evaluation baseline only; it is not compatible with the <2ms OSS-FX shipping path.
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** (Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12; arXiv:2605.11489) — Couples gradient-aware SR with a compact U-Net frame-interpolation module over 3DGS renders. **Why it matters to OSS:** closest published SR+FG Gaussian competitor, but it retains separate SR and FI modules; retain it as the two-network baseline for the one-canvas claim.

### Inference Optimization
- No fresh arXiv optimization result met the direct-runtime bar. **Practical lead:** AMD’s official RDNA4 WMMA guidance (captured previously) makes wide-K FP8/INT8 fusion a concrete HIP kernel experiment, but only after #9 establishes the student control and #8 has a real HIP scaffold.

## Archive.org Findings
- Advanced Search remains non-recovering/noisy: `"gaussian splatting" super-resolution` returned 3 archived GitHub-star lists; `"frame generation" deep learning` returned 3 results, mostly non-paper material; `"real-time upscaling" game` returned 0.
- Targeted names did not recover a removed primary: GS-STVSR 0, GSASR 1 unrelated CIA document, `GRAPE gaussian` 6 acronym/noise hits, and `STSS super-resolution` 0. This is **not** evidence that any paper was taken down.

## GitHub / Open Source
- GitHub REST searches since 2026-07-08 returned **0** Gaussian-SR repositories and **0** DLSS/FSR/XeSS analysis tools. `frame generation open source` returned one irrelevant zero-star pentest project; `neural upscaler` is dominated by zero-star GAN utilities/copies.
- No credible new OSS competitor or analysis tool was verified. NanoVSR remains the actionable open-source baseline (MIT implementation/ONNX exporter), already tracked.

## News

### NVIDIA / DLSS
- **Verified, competitive threat:** NVIDIA’s DLSS 4.5 page documents Dynamic Multi Frame Generation and up to 6× MFG on RTX 50, plus a second-generation transformer SR model. Its dynamic controller varies the multiplier to target display refresh/frame-rate goals. OSS should compare base-frame latency, artifacts, and engine-input cost—not headline generated FPS—and preserve its no-second-network argument.

### AMD / FSR
- **Verified:** AMD’s FSR SDK 2.3 (2026-06-24) brings ML FSR Upscaling 4.1.1 to RDNA3/RX 7000, while Frame Generation 4.0.1 improves motion-vector/camera processing in corner cases. **Impact:** RDNA3 is now an immediate cross-vendor quality/integration target; the MV/camera contract is directly comparable with OSS capture inputs.
- **RSS-only lead:** reports of 8× FSR MFG remain unconfirmed by AMD. Monitor; do not plan around it.

### Intel / Apple / Engines
- **Verified platform change:** Intel’s Arc G-series handheld announcement positions XeSS 3 as SR + Multi-Frame Generation + Xe Low Latency on Xe3/B390-class handhelds. **Impact:** reinforces the Pico handheld latency target and the need for a genuine Intel backend, but yields no implementable API/performance datum.
- No primary Apple MetalFX or Unreal/Unity upscaling-specific announcement was found in the 14-day window. Unity 6.7/Unity 7 discussion results are not sufficient evidence of a new upscaling feature.

## Action Items / Follow-ups
- [ ] **CRITICAL:** recover/inspect the 3080 Ti trainer. It is still step 35,200/loss 0.10166/viz 35,077 after **144h**; healthy publishing/CDN does not demonstrate training health.
- [ ] Deep-read SPARE-GS; run a fixed-active-canvas ablation of global regional marginal-utility quotas versus current parent/child loss-adaptive density, with temporal-artifact gates.
- [ ] Make NanoVSR the #9 reparameterized-CNN control and measure TensorRT first, then HIP, with actual OSS G-buffer/MV inputs.
- [ ] Add SPIN-4DGS fast-motion/disocclusion tests before changing `V_xt`; test residual attribute prediction only where exact MV warp confidence is low.
- [ ] Treat DLSS 4.5 Dynamic/6× MFG and XeSS 3 MFG as positioning pressure; benchmark OSS-FX on input latency and error, not generated-frame multiplier.

# OSS Research Brief — 2026-08-04

_Collection: local Git/README, GitHub REST, rendered public dashboard, targeted arXiv Atom queries, Archive.org Advanced Search, GitHub repository APIs, and Google News RSS. RSS-only items are leads, not vendor-confirmed announcements._

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated dashboard/API training controls. The preceding work preserves the v7 step-5K snapshot and recovery instructions, fixes cross-OS checkpoint resume and WSL2 DataLoader deadlock behavior, and adds FlashAttention 2, TF32 convolution, and prefetch throughput work.
- **Open issues:** 12: #14 VRAM footprint, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 decoupled G-buffer encoder, #9 reparameterized student, #8 cross-vendor kernels, #7 FSR 4 teacher signal, #6 Pico launch, #5 no-ML variant, #4 <=1M student distillation, #3 counting-sort tile bin, #1 low-rank rasterizer.
- **README snapshot:** `srcnn-v7.0-pico-005` is the HAT-Tiny research teacher: N-D `(x,y,t)` Gaussians, `V_xt` motion covariance, loss-adaptive density, Mip-Splat filtering, and alpha=0.5 OSS-FX supervision. The shipping <=1M TensorRT-FP8 student and non-CUDA backends remain unbuilt. Published held-out quality is still v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-08-04, refreshed):** headline says v7 is “training right now,” `36,650 / 100,000`, loss `0.19268`, viz `36,000`, ETA ~1.1d and “live.” Health separately says **Trainer offline, stuck at step 36,650 for 71h**; watcher published 0s ago, CF Worker HTTP 200 in 98ms, R2 origin HTTP 200, and GPU stats are unavailable. Step/loss did not move while the stuck counter advanced 59h→71h. The publishing path is healthy; trainer/metric emission is not. “Live” is stale/conflicting dashboard copy, not evidence of recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis** — Zhengyang Zhang, Ziyu Lu, PengCheng Li, Hongbo Duan, Yi Liu, Pengting Luo, Peiyu Zhuang, Xinghui Li, Shaohua Ma; 2026-08-03; [arXiv:2608.01958](https://arxiv.org/abs/2608.01958). It decomposes dynamic-GS motion into Fourier components and regularizes frequencies to retain complex motion while suppressing jitter/drift. **Why it matters to OSS:** direct ablation lead for whether v7 `V_xt` needs a residual multi-frequency motion parameterization or frequency-aware temporal regularizer on difficult motion; preserve exact engine MVs as the baseline.
- **DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization** — Minnan Pei et al.; 2026-08-03; [arXiv:2608.02099](https://arxiv.org/abs/2608.02099). It decouples alpha/transmittance checks from blending, repacks irregular tile work into dense stages, and reports 1.82–6.02x end-to-end speedup in its accelerator evaluation. **Why it matters to OSS:** strongest new raster-kernel lead for #3/#12/#13—profile the proposed parse/reorganize/blend split against OSS’s tile bin and packed canvas before treating its hardware claims as transferable.
- **ASTRA: Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment** — Junyu Zhu, Hao Zhu, Xinzhuo Zhang, Hongdong Li, Zhan Ma, Xun Cao; 2026-08-03; [arXiv:2608.02006](https://arxiv.org/abs/2608.02006). It aligns reconstruction motion with explicit 2D trajectories and reliability masks rather than only photometric residuals. **Why it matters to OSS:** a training-loss/control reference for detecting bad/stale engine MVs and disocclusions; use only if exact-MV quality diagnostics expose a gap.

### Super-Resolution / Upscaling

- **CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution** — Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03; [arXiv:2608.02206](https://arxiv.org/abs/2608.02206). It treats LR observations as an anchor, applies evidence-conditioned correction to conflicting HR priors, and routes high-frequency gradients/densification by patch reliability. **Why it matters to OSS:** direct training ablation for v7 parent-child density: gate densification and teacher/perceptual gradients by LR/MV/depth confidence so hallucinated detail does not override engine evidence.
- No newly published material game-SR or <=1M student-distillation paper cleared the shipping relevance gate. Already-tracked NanoVSR remains the compact reparameterized-CNN control for #9/#4; its official MIT repo was active on 2026-08-04.

### Frame Extrapolation / Generation

- No new material single-network, exact-motion-vector extrapolation paper. Broad results were world-model/diffusion/video-generation work rather than a credible <2ms OSS-FX competitor. OSS’s differentiation remains an architectural claim until alpha-time output is evaluated on held-out future frames.

### Inference Optimization

- **DeGS** above is the material new optimization result. Its key lesson is workload organization, not a drop-in design: benchmark end-to-end raster time, L2/DRAM traffic, active-Gaussian count, and quality before altering #3 or #12.
- No new material TensorRT/INT8-SR/kernel-fusion paper; existing QIRF and NanoVSR references remain higher-priority controls.

## Archive.org Findings

- Advanced Search found three archived GitHub/star-list snapshots for `"gaussian splatting" super-resolution`; no validated paper copy. `"frame generation" deep learning` returned FluidFrames/DLSS Swapper builds and an unrelated diffusion record; `"real-time upscaling" game` returned zero.
- Known-work probes remain noisy/unhelpful: GS-STVSR and STSS returned zero; GSASR returned an unrelated 1961 CIA item; GRAPE results were unrelated. No archived version of GS-STVSR, GSASR, GRAPE, or STSS was verified in this pass. This is not evidence that none exists.

## GitHub / Open Source

- No credible new direct OSS competitor or open-source DLSS/FSR analysis tool found. [3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) is Apache-2.0 but medical CT SR (4 stars; pushed 2026-07-28), not a game/canvas runtime baseline.
- [NanoVSR](https://github.com/filippawlicki/nanovsr) is now 35 stars, MIT, and pushed 2026-08-04; it is the practical reference implementation to inspect for deploy-graph/reparameterization tests, not a competitive game integration threat.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains a GPL-3.0, RDNA3/G-buffer data-contract lead (15 stars; pushed 2026-07-24). Inspect interfaces/evaluation only; maintain GPL isolation.
- [OptiScaler](https://github.com/optiscaler/OptiScaler) remains the integration UX benchmark (9,635 stars / 351 forks; pushed 2026-08-03). Its multi-upscaler/FG bridge is a compatibility threat only, not an OSS-equivalent unified reconstruction architecture; GPL-3.0 also requires isolation.

## News

- **Intel lead (RSS-only):** a 2026-08-03 report says the XeSS plugin 3.1 adds UE 5.8 support. Treat as an SDK compatibility watch until verified on Intel’s release channel; if confirmed, it raises the integration baseline for OSS’s UE path, not the underlying architecture bar.
- **DLSS/FSR:** 7–14 day RSS results were chiefly benchmarks, game patches, and unverified “DLSS 4.5/5” headlines. No new primary NVIDIA/AMD release was verified in this pass; do not move architecture based on them.
- **Apple/Metal:** no qualifying primary Apple Silicon/Metal upscaling update was found. One RSS result was low-quality/noise and discarded.
- **GPU/engine:** no material primary NVIDIA/AMD/Intel GPU or UE/Unity upscaling announcement was verified. Reuters’ AMD AI-server production item is not a game-upscaling signal.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or its metrics emitter; require a higher step, fresh GPU telemetry, and a new viz before labeling v7 live.
- [ ] Run a small **CLEAR-style confidence-gated densification** ablation: compare current loss-adaptive parent-child allocation with LR/MV/depth-confidence gating and measure detail vs temporal artifacts.
- [ ] After #13, prototype/benchmark the **DeGS workload split** around the existing counting-sort tile bin: record raster latency, L2 hit rate, DRAM bytes, occupancy, and image parity.
- [ ] Stress-test **FAST-GS-inspired frequency-aware temporal regularization** only on rapid/nonlinear motion; preserve exact engine MVs and reject it if it adds drift or runtime state.
- [ ] Track Intel XeSS UE 5.8 support from an Intel primary release; keep OptiScaler as the DLL-shim UX comparator with GPL isolation.

## Bibliography Update

- Added CLEAR, DeGS, and FAST-GS: all three are new, primary-metadata-verified architecture/training/runtime references. ASTRA is memo-only because it is a diagnostic/loss analogue rather than a direct shipping or canvas-design result.

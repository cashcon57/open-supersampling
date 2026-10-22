# OSS Research Brief — 2026-08-05

_Collection: local Git/README, GitHub REST, rendered public dashboard, arXiv recent-category pages (the Atom API returned HTTP 429 during arXiv's Aug 4–5 maintenance), Archive.org Advanced Search, GitHub repository APIs, direct vendor pages, and Google News RSS. RSS-only items are leads, not vendor-confirmed announcements._

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add tailnet-gated training controls/API to the dashboard. Immediately prior: preserved v7 step-5K recovery, fixed WSL2 DataLoader and cross-OS resume issues, and added FlashAttention 2/TF32/prefetch throughput work.
- **Open issues (12):** #14 VRAM footprint; #13 Nsight L2 profile; #12 packed low-rank canvas; #10 decoupled G-buffer encoder; #9 reparameterized student; #8 non-CUDA kernels; #7 FSR 4 teacher signal; #6 Pico launch; #5 no-ML variant; #4 <=1M student; #3 counting-sort tile bin; #1 low-rank rasterizer.
- **README snapshot:** `srcnn-v7.0-pico-005` remains the HAT-Tiny research teacher: N-D `(x,y,t)` Gaussians, `V_xt` motion covariance, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The <=1M TensorRT-FP8 student and non-CUDA backends remain unbuilt. Published held-out quality is still v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-08-05):** headline says v7 is “training right now,” `36,650 / 100,000`, loss `0.19268`, viz `36,000`, ETA ~1.1d, and “live.” Health separately says **Trainer offline, stuck at step 36,650 for 10h**; watcher published just now, CF Worker HTTP 200 in 99ms, R2 origin HTTP 200, and GPU stats are unavailable. Step/loss did not advance since the 2026-08-04 brief, but the displayed stuck counter reset from 83h to 10h with no metric progress—a dashboard health-counter inconsistency, not recovery. The public publishing path is healthy; trainer/metric emission is not. “Live” remains stale/conflicting copy, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

_No new material record after the 2026-08-04 brief. arXiv Atom was rate-limited during a posted maintenance window; recent cs.CV/cs.GR/cs.MM listing pages were inspected as the primary fallback and did not expose a newly material match in their current entries. The following recent papers remain the relevant items from the targeted pass._

- **[FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis](https://arxiv.org/abs/2608.01958)** — Zhengyang Zhang, Ziyu Lu, PengCheng Li, Hongbo Duan, Yi Liu, Pengting Luo, Peiyu Zhuang, Xinghui Li, Shaohua Ma; 2026-08-03. Fourier motion components plus frequency-aware regularization improve complex dynamic-GS motion and long-term coherence. **Why it matters to OSS:** ablate a residual frequency-aware temporal regularizer only on nonlinear/rapid motion; exact engine MVs remain the primary motion signal and drift-free baseline.
- **[DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization](https://arxiv.org/abs/2608.02099)** — Minnan Pei, Gang Li, Zeyu Zhu, Siting Wang, Junwen Si, Zhuoran Song, Yu Feng, Fangxin Liu, Xiaoyao Liang, Jian Cheng; 2026-08-03. It separates alpha/transmittance checking from blending, packs irregular splat work into dense stages, and reports 1.82–6.02× end-to-end accelerator speedups. **Why it matters to OSS:** strongest kernel-design lead for #3/#12/#13; benchmark its parse/reorganize/blend decomposition against the existing tile bin before assuming its ASIC results transfer to CUDA/HIP.
- **[ASTRA: Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment](https://arxiv.org/abs/2608.02006)** — Junyu Zhu, Hao Zhu, Xinzhuo Zhang, Hongdong Li, Zhan Ma, Xun Cao; 2026-08-03. Explicit 2D trajectory alignment and reliability masks improve reconstruction under temporal offsets. **Why it matters to OSS:** a diagnostic/loss analogue for stale or bad engine-MV/disocclusion regions—not a shipping-path replacement for exact MVs.

### Super-Resolution / Upscaling

- **[CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution](https://arxiv.org/abs/2608.02206)** — Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03. It anchors optimization in trusted LR evidence while routing high-frequency gradients and densification using patch reliability. **Why it matters to OSS:** direct ablation for v7 parent-child density—gate densification/teacher gradients using LR, MV, and depth confidence so hallucinated detail cannot override engine evidence.
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** — Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11. A structurally reparameterized fully convolutional VSR model deploys as standard convolutions; its 644k model reports 28.64 dB on REDS4 and 27.2 FPS on Jetson Orin NX. **Why it matters to OSS:** already tracked, but remains the clearest practical control for #9/#4 deployment-graph and reparameterization tests; it is not a direct game/MV/canvas competitor.
- **[TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution](https://arxiv.org/abs/2607.22231)** — Sicheng Gao, Zhuyun Zhou, Yixuan Liu, Tong Shen, Zongwei Wu, Radu Timofte; 2026-07-24. It adaptively routes/merges tokens in diffusion VSR. **Why it matters to OSS:** useful selective-compute framing, but not applicable to the <2ms student path; do not prioritize over packed canvas/raster work.

### Frame Extrapolation / Generation

- No credible new single-network, exact-motion-vector, real-time extrapolation paper appeared. Broad “frame generation” hits were diffusion/world-model work, not a <2ms OSS-FX competitor. OSS’s differentiation remains unvalidated until α-time output receives held-out future-frame evaluation.

### Inference Optimization

- **DeGS** is the only material new optimization result. Its transferable hypothesis is dataflow/workload organization, not a drop-in kernel: profile raster latency, active-Gaussian count, L2 hit rate, DRAM bytes, occupancy, and image parity first.
- No material newly published TensorRT/INT8-SR/kernel-fusion paper cleared the relevance gate. NanoVSR and existing QIRF references remain better student controls.

## Archive.org Findings

- Advanced Search found three archived GitHub/star-list snapshots for `"gaussian splatting" super-resolution`; no validated paper copy. `"frame generation" deep learning` returned FluidFrames/DLSS Swapper builds plus unrelated diffusion; `"real-time upscaling" game` returned zero.
- Known-work probes are still noisy: GS-STVSR and STSS returned zero; GSASR returned an unrelated 1961 CIA record; GRAPE results were unrelated. No archived GS-STVSR, GSASR, GRAPE, or STSS version was verified. This is not evidence of absence.

## GitHub / Open Source

- No credible new direct OSS competitor or new open DLSS/FSR analysis tool. [3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) is Apache-2.0 but medical CT SR (4 stars; pushed 2026-07-28), not a game/canvas baseline.
- [NanoVSR](https://github.com/filippawlicki/nanovsr) remains the practical open reparameterized-CNN reference for student inference (MIT; arXiv authors publish it as the code release).
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains a GPL-3.0 RDNA3/G-buffer data-contract lead (15 stars; pushed 2026-07-24): inspect interfaces/evaluation only; keep GPL isolation.
- [OptiScaler](https://github.com/optiscaler/OptiScaler) remains the DLL-shim integration UX benchmark rather than an OSS-equivalent reconstruction architecture; maintain GPL isolation.

## News

- **Primary vendor check:** no new 7–14-day NVIDIA/AMD/Intel/Apple/Unreal/Unity architecture announcement was verified. NVIDIA's DLSS 4.5 page is a CES announcement, not a new update; it confirms Dynamic MFG/6× MFG and a second-generation SR transformer but supplies no fresh runtime delta. AMD's FSR UE guide was last updated March 12 and supports UE 5.2–5.7: ML SR + ML FG on RDNA 4, analytical fallbacks on RDNA 3.5/older, and an explicit warning that missing WPO-object velocities cause blur/ghosting. That remains a useful OSS capture/integration data-contract reference, not current news.
- **Intel lead (RSS-only):** VideoCardz reported on 2026-08-03 that XeSS plugin 3.1 adds UE 5.8 support. Treat it as an SDK-compatibility watch pending Intel primary release notes; if confirmed, it raises the UE integration baseline, not the OSS architecture bar.
- **DLSS/FSR RSS:** results were game patches, benchmarks, and unverified “DLSS 4.5/5” headlines. A reported Steam Machine/Proton FSR 4.1 item likewise lacked a primary Valve/AMD source in this pass. Do not move architecture based on these leads.
- No material primary GPU announcement or UE/Unity upscaling update was found; Apple/Metal had no qualifying recent result.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or its metrics emitter; require a higher step, fresh GPU telemetry, and a new viz before calling v7 live.
- [ ] Run a small **CLEAR-style confidence-gated densification** ablation against current loss-adaptive parent-child allocation; measure detail recovery and temporal artifacts.
- [ ] After #13, prototype/benchmark the **DeGS workload split** around the counting-sort tile bin: raster latency, L2 hit rate, DRAM bytes, occupancy, and image parity.
- [ ] Stress-test **FAST-GS-style frequency-aware temporal regularization** only on difficult nonlinear motion; reject it if it introduces drift or extra runtime state.
- [ ] Verify the XeSS UE 5.8 claim through Intel primary release notes; retain OptiScaler as the GPL-isolated DLL-shim UX comparator.

## Bibliography Update

- No append: CLEAR, DeGS, FAST-GS, and NanoVSR were already recorded by the 2026-08-04 monitor. This pass surfaced no new material paper that cleared the stricter bibliography relevance gate.

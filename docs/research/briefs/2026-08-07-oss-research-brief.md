# OSS Research Brief — 2026-08-07

_Collection: local Git/README, rendered public dashboard, GitHub REST, arXiv recent-category pages (Atom search was rate-limited), Archive.org Advanced Search/Wayback CDX, vendor primary pages, and Google News RSS. “New” below means newly material in the current monitoring window; the four papers listed were already captured in the bibliography by the Aug 3–6 runs, so no duplicate bibliography entry is warranted._

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add a tailnet-gated training-control API and dashboard panel. The preceding commits preserve the v7 step-5K recovery snapshot, fix WSL2 DataLoader/cross-OS checkpoint resume, and add FlashAttention 2, TF32 convolution, and prefetch throughput work.
- **Open issues:** 12 open issues. Near-term execution blockers remain #14 (v6.2 VRAM measurement), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #3 (tile bin), #4/#9 (student distillation/reparameterization), and #8 (cross-vendor kernels).[2]
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is the active HAT-Tiny teacher: native `(x,y,t)` Gaussians, `V_xt` covariance motion, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The ≤1M TensorRT-FP8 shipping student and non-CUDA runtimes remain unbuilt.[3]
- **Training dashboard:** headline copy still says “training right now,” but the run is unchanged at **36,650 / 100,000**, loss **0.19268**, viz **36,000**. The Trainer health tooltip says **offline, stuck at step 36,650 for 58h**; Watcher is healthy (published 1s ago), CF Worker healthy (HTTP 200/101ms), and R2 origin healthy. GPU stats are unavailable. This is an operational contradiction: the publishing path is live, but training is not advancing.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields** (arXiv:2608.04581; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko; 2026-08-05; already tracked) — Allocates both Gaussian cardinality and feature-channel capacity to local spatial-temporal complexity. **Why it matters to OSS:** strongest direct reference for v7 parent-child density and #12 packed-canvas budgeting; test adaptive child count *and* feature width under one fixed memory budget.[13]
- **FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis** (arXiv:2608.01958; Zhengyang Zhang, Ziyu Lu, PengCheng Li, Hongbo Duan, Yi Liu, Pengting Luo, Peiyu Zhuang, Xinghui Li, Shaohua Ma; 2026-08-03; already tracked) — Uses Fourier motion components and frequency-aware regularization for high-frequency motion and long-horizon coherence. **Why it matters to OSS:** teacher-side hard-motion regularizer ablation only; exact engine MVs remain OSS’s stronger primary signal.[14]
- **Current scan result:** direct arXiv Atom queries for all requested term families returned HTTP 429. Recent pages for cs.CV/cs.GR/cs.MM were inspected instead; the Aug 7 listings did not surface another material game-SR, 2D-Gaussian-SR, native-time 4D-GS, or low-latency frame-generation paper beyond the already-tracked set.[4][5][6]

### Super-Resolution / Upscaling

- **CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution** (arXiv:2608.02206; Hantang Li, Qiang Zhu, Xiandong Meng, Debin Zhao, Xiaopeng Fan; 2026-08-03; already tracked) — Routes high-frequency gradients and densification by evidence/reliability instead of uniformly trusting weak observations. **Why it matters to OSS:** direct lead for confidence-gated v7 densification when LR color, depth, exact MVs, and teacher/perceptual signals conflict.[15]
- No new material real-time game-SR or ≤1M student paper cleared the relevance gate. NanoVSR and the existing 2D-Gaussian SR work remain the practical student comparison set.

### Frame Extrapolation / Generation

- No credible new single-network, exact-engine-MV, <2ms frame-extrapolation competitor appeared. Broad recent results were diffusion/world-model work, which may be useful as offline perceptual teachers but do not alter OSS-FX’s shipping architecture decision.

### Inference Optimization

- **DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization** (arXiv:2608.02099; Minnan Pei, Gang Li, Zeyu Zhu, Siting Wang, Junwen Si, Zhuoran Song, Yu Feng, Fangxin Liu, Xiaoyao Liang, Jian Cheng; 2026-08-03; already tracked) — Separates alpha/transmittance parsing from blending and reorganizes irregular splat work before blending. **Why it matters to OSS:** strongest current hypothesis for #3/#12/#13; evaluate it on raster latency, L2 hit rate, DRAM bytes, occupancy, and image parity across CUDA/HIP rather than accepting accelerator-specific headline numbers.[16]
- No material new TensorRT/INT8-SR/kernel-fusion result surfaced in the fallback scan.

## Archive.org Findings

- Advanced Search was noisy: `"gaussian splatting" "super-resolution"` returned three archived GitHub/star-list snapshots, while frame-generation and real-time-game-upscaling queries were dominated by unrelated media/software records. Treat this as inventory coverage, not evidence of absence.[7]
- Targeted Wayback CDX confirms GSASR GitHub snapshots, including 2025-08-04 and 2026-06-07. The direct GS-STVSR lookup timed out, GaussianSR returned HTTP 503, and the GRAPE wildcard returned no snapshots; none is sufficient to claim removal or nonexistence.[8]

## GitHub / Open Source

- No newly created direct OSS competitor or useful open DLSS/FSR-analysis tool surfaced. Keyword hits were mostly off-domain or zero-star projects.
- **RDNU** remains the meaningful active comparison: public GPL-3.0 RDNA3-focused G-buffer-guided neural upscaler, 15 stars, updated/pushed 2026-07-24, with a validated WMMA INT8 1×1-convolution phase. **Positioning:** inspect its G-buffer contract and benchmark methodology; do not reuse GPL code.[9]
- **Intel XeSS:** Intel’s public XeSS 3 SDK repository was updated to SDK 3.0.2 on July 23. It exposes separate XeSS-SR, XeSS-FG, and XeLL components, and links XeSS Inspector for API/event/frame-dump inspection. **Opportunity:** use the Inspector data-contract vocabulary as a validation checklist for the eventual DLL shim; it also reinforces OSS’s one-canvas SR+FX positioning against split vendor components.[10]

## News

- **NVIDIA / DLSS:** NVIDIA’s July 28 update says Halo: Campaign Evolved has DLSS Super Resolution plus Frame Generation, while Palworld 1.0 and Killing Floor 3 can be upgraded to DLSS 4.5 through the NVIDIA app. This is rollout news, not a new algorithm disclosure, but it continues the proprietary separated-SR/FG baseline OSS targets.[11]
- **AMD / FSR:** AMD’s current FSR Upscaling page lists SDK 2.3 / FSR Upscaling 4.1.1, adds RDNA3 discrete-GPU support, and says the UE plugin remains at 4.0.3 for UE 5.2–5.7 pending update. AMD also states the ML path expects color, depth, and screen-space motion vectors. **Opportunity:** OSS’s engine-buffer contract aligns with the incumbent integration surface; the delayed UE plugin version is an integration/positioning gap, not proof of performance superiority.[12]
- **Intel / Apple / engines:** Google News RSS produced a headline-level XeSS UE 5.8-plugin lead but no corresponding Intel primary confirmation. No material Apple Metal/MetalFX, Unity, or GPU-launch change was verified this window; keep these on watch rather than promote them to roadmap facts.

## Action Items / Follow-ups

- [ ] **Critical:** restore the 3080 Ti trainer or its metric emitter; require a higher step, fresh GPU telemetry, and visualization before trusting “live training” copy.
- [ ] Deep-read ACA-GS and run a fixed-memory ablation: adaptive child count + feature-width mask versus current parent-child density; score temporal PSNR/LPIPS, splat count, and bytes/frame.
- [ ] Prototype CLEAR-style reliability gates across LR color, depth, and exact-MV signals for densification and teacher-gradient routing.
- [ ] After #13, prototype a DeGS-like parse/reorganize/blend tile path; reject it unless it improves measured latency/bandwidth without image drift.
- [ ] Add XeSS Inspector’s required input/event/frame-dump checks to the Sprint-7 DLL-shim validation plan; monitor AMD’s UE 4.1.1 plugin update.

## Sources

[1] https://opensupersampling.org
[2] https://github.com/cashcon57/open-supersampling/issues
[3] https://github.com/cashcon57/open-supersampling
[4] https://arxiv.org/list/cs.CV/recent
[5] https://arxiv.org/list/cs.GR/recent
[6] https://arxiv.org/list/cs.MM/recent
[7] https://archive.org/advancedsearch.php
[8] https://web.archive.org/cdx/search/cdx?url=github.com%2FChrisDud0257%2FGSASR
[9] https://api.github.com/repos/Adam-AtlasSoftware/RDNU
[10] https://github.com/intel/xess
[11] https://www.nvidia.com/en-us/geforce/news/halo-campaign-evolved-mistfall-hunter-corsair-cove-dlss
[12] https://gpuopen.com/amd-fsr-upscaling
[13] https://arxiv.org/abs/2608.04581
[14] https://arxiv.org/abs/2608.01958
[15] https://arxiv.org/abs/2608.02206
[16] https://arxiv.org/abs/2608.02099

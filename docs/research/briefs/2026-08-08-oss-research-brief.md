# OSS Research Brief — 2026-08-08

_Collection: local Git/README, rendered public dashboard, GitHub REST, arXiv Atom API, Archive.org Advanced Search/Wayback CDX, vendor primary pages, and Google News RSS. “New” means newly material in this monitoring window; low-relevance hits were filtered rather than treated as OSS leads._

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add a tailnet-gated training-control API/dashboard panel. The preceding commits preserve the v7 recovery checkpoint; remove WSL2 DataLoader and cross-OS resume friction; and add FlashAttention 2, TF32 convolution, and prefetch throughput work.
- **Open issues:** 12 open issues. The immediate architecture/performance chain remains #14 (v6.2 VRAM measurement), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #3 (tile bin), then student distillation/reparameterization (#4/#9) and cross-vendor kernels (#8).[2]
- **README snapshot:** v7 `srcnn-v7.0-pico-005` remains a HAT-Tiny research teacher with native `(x,y,t)` Gaussians, `V_xt` covariance motion, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The ≤1M TensorRT-FP8 shipping student and non-CUDA runtimes are still unbuilt.[3]
- **Training dashboard (refreshed):** headline copy says training is live at **36,650 / 100,000**, loss **0.19268**, viz **36,000**, but the Trainer tooltip says **offline, stuck at step 36,650 for 82h**. Watcher is healthy (published 1s ago), CF Worker is healthy (HTTP 200/108ms), and R2 origin is healthy; GPU stats are unavailable. Publishing is alive but training has not advanced—the dashboard’s “live” label is misleading operationally.[1]

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- No newly submitted material Gaussian-temporal paper cleared the OSS relevance gate. Atom results re-surfaced **ACA-GS** (arXiv:2608.04581), **DeGS** (2608.02099), and **FAST-GS** (2608.01958), all already tracked in `BIBLIOGRAPHY.md`; their adaptive capacity, workload reorganization, and frequency-aware temporal motion remain the strongest v7 density/rasterization leads.[4]

### Super-Resolution / Upscaling

- **A Multi-Layer System for Ultra-High-Resolution Static 360-Degree Telepresence** (arXiv:2608.05570; Jiapeng Chi, Gerd Bruder, Carsten Neumann, Carolina Cruz-Neira, Dirk Reiners; 2026-08-06) — a systems paper on ultra-high-resolution *static* 360° telepresence. **Why it matters to OSS:** low-priority reference for multi-layer resolution/bandwidth budgeting only; it has no temporal game reconstruction, Gaussian canvas, or ≤2ms student contribution and does **not** warrant a bibliography addition.[5]
- Atom also re-surfaced **NanoVSR** (arXiv:2607.10495), **TRaM-VSR** (2607.22231), and two arbitrary-scale Gaussian-SR papers (2606.29453/2606.29400). They are already catalogued; no new real-time game-SR or student-distillation paper changed the training plan.[6]

### Frame Extrapolation / Generation

- No credible new single-network, exact-engine-MV, real-time frame-extrapolation competitor surfaced. The newest direct query hit, **Kastor** (arXiv:2608.06107; Guillaume Couairon, Alexis Jacq, Yu-Han Wu, Renu Singh, Yana Hasson, Quentin Berthet, Romuald Elie; 2026-08-06), is PDE-simulation emulation rather than visual frame generation and was filtered out.[7]
- Existing **SPEED** (one-step diffusion VFI) and **MASS** (motion-aligned selective scan) remain useful offline/high-motion teacher or benchmark comparators—not shipping alternatives to OSS-FX.[8]

### Inference Optimization

- No material new TensorRT/INT8-SR/kernel-fusion paper appeared. The Atom result **TileSight** (arXiv:2607.22432) is a general tile-centric GPU performance model, not an upscaler, but is a useful methodology reference for issue #13: measure tile occupancy, L2 behavior, DRAM traffic, and imbalance before choosing the packed-canvas layout.[9]

## Archive.org Findings

- Advanced Search remains noisy: `"gaussian splatting" "super-resolution"` returned three archived GitHub/star-list snapshots; `"frame generation" "deep learning"` returned a diffusion registration plus FluidFrames/DLSS-Swapper binaries; and `"real-time upscaling" game` returned zero. This is inventory coverage, not evidence of absence.[10]
- Targeted Wayback CDX finds GSASR GitHub snapshots (including 2025-03-13, 2025-08-01, and 2025-08-04). The broad GS-STVSR/GRAPE/STSS wildcard lookups returned no records; that is insufficient to claim a project/paper is gone.[11]

## GitHub / Open Source

- No new direct OSS competitor or open DLSS/FSR-analysis tool passed relevance review. Searches were dominated by unrelated zero-star projects and off-domain medical SR.
- **RDNU** remains the meaningful active comparison: 15-star GPL-3.0 RDNA3-focused, G-buffer-guided neural upscaler, last pushed July 24. Inspect its G-buffer contract and benchmarks, but do not reuse GPL code.[12]
- **Intel XeSS SDK 3.0.2** was released July 24; its visible changes are XeLL stability/interop fixes (including a non-Intel memory leak and Streamline proxy handling), not a new reconstruction architecture. Its separate XeSS-SR/XeSS-FG/XeLL API remains a useful contrast for OSS’s one-canvas SR+FX positioning and validation-contract checklist.[13]
- `julesklord/bananascaler` is a newly active (Aug 4), MIT, zero-star GPU video-upscaler TUI. Watch only; its public metadata gives no evidence of game integration, temporal quality, or production inference work.[14]

## News

- **NVIDIA / DLSS:** NVIDIA’s July 28 update is rollout news: *Halo: Campaign Evolved* supports DLSS Super Resolution and Frame Generation; *Palworld 1.0* and *Killing Floor 3* can be upgraded to DLSS 4.5 through the NVIDIA app. No architecture disclosure this window.[15]
- **AMD / FSR:** AMD’s current primary page lists FSR Upscaling 4.1.1 in FSR SDK 2.3, adds RDNA3 discrete-GPU support, and says the UE plugin remains at 4.0.3 pending a 4.1.1 update. The ML path’s claims—less moving-object ghosting, particle preservation, and temporal stability—make OSS’s exact-MV/disocclusion evaluation suite more important, not less.[16]
- **Intel / engines:** Google News RSS has a 3-Aug headline-level lead that an Intel XeSS 3.1 UE plugin adds UE 5.8 support, but no Intel primary source was located; do not promote it to roadmap fact.[17]
- **Apple / GPU announcements / Unity:** no material primary Apple MetalFX, Unity-upscaling, or GPU-platform update was verified during the 7–14 day scan. RSS claims outside the above were low-quality, non-primary, or unrelated.

## Action Items / Follow-ups

- [ ] **Critical:** restore the 3080 Ti trainer/metric emitter; require an increased step, fresh GPU telemetry, and a new visualization before treating the v7 run as live.
- [ ] Complete #14 → #13 → #12 in order; gate packed canvas choices on measured bytes/frame, L2 hit rate, occupancy, tile imbalance, and image parity—not only theoretical 32B/Gaussian accounting.
- [ ] Keep ACA-GS/CLEAR as the density-ablation set: compare fixed-memory adaptive child count plus feature-width/reliability gating against current parent-child density.
- [ ] Use AMD’s FSR 4.1.1 ghosting/particle claims as external test classes: add disocclusion, moving-particle, thin-specular, and exact-MV confidence slices to OSS-FX evaluation.
- [ ] Watch Intel’s UE 5.8-plugin lead until a primary Intel release appears; retain XeSS Inspector/API event/frame-dump concepts for the future DLL-shim validation plan.

## Sources

[1] https://opensupersampling.org
[2] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[3] https://github.com/cashcon57/open-supersampling
[4] https://export.arxiv.org/api/query?search_query=%28abs%3A%22gaussian%20splatting%22%20AND%20%28abs%3Atemporal%20OR%20abs%3Adynamic%20OR%20abs%3A4D%29%29&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending
[5] https://arxiv.org/abs/2608.05570
[6] https://export.arxiv.org/api/query?search_query=%28abs%3A%22video%20super-resolution%22%20OR%20abs%3A%22real-time%20super%20resolution%22%20OR%20abs%3A%22neural%20rendering%22%20AND%20abs%3A%22super-resolution%22%29&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending
[7] https://arxiv.org/abs/2608.06107
[8] https://arxiv.org/abs/2607.15585
[9] https://arxiv.org/abs/2607.22432
[10] https://archive.org/advancedsearch.php
[11] https://web.archive.org/cdx/search/cdx?url=github.com%2FChrisDud0257%2FGSASR%2A&output=json
[12] https://api.github.com/repos/Adam-AtlasSoftware/RDNU
[13] https://github.com/intel/xess/releases/tag/v3.0.2
[14] https://api.github.com/repos/julesklord/bananascaler
[15] https://www.nvidia.com/en-us/geforce/news/halo-campaign-evolved-mistfall-hunter-corsair-cove-dlss
[16] https://gpuopen.com/amd-fsr-upscaling
[17] https://news.google.com/rss/search?q=Intel+XeSS+when%3A14d

# OSS Research Brief — 2026-07-31

_Collection: local Git/README, GitHub REST, rendered training dashboard, arXiv Atom/direct abstract pages, Archive.org Advanced Search/Wayback, GitHub repository metadata, a primary NVIDIA post, and Google News RSS. RSS-only items are labelled as leads._

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add tailnet-gated training controls (dashboard, API, and control panel). Recent training resilience/performance work preserves the v7 step-5K recovery snapshot, fixes Linux resume of Windows checkpoints and WSL2 loader deadlock, and adds direct FlashAttention 2 plus TF32/prefetch improvements.
- **Open issues:** 12. The near-term technical gates are #14 (v6.2 VRAM measurement), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #9 (reparameterized student), #8 (cross-vendor kernels), #7 (FSR 4 teacher signal), and #4 (≤1M student distillation).
- **README snapshot:** active `srcnn-v7.0-pico-005` is the HAT-Tiny research teacher using N-D `(x,y,t)` Gaussians, `V_xt` motion covariance, parent-child density, Mip-Splat filtering, and alpha=0.5 OSS-FX supervision. The ≤1M TensorRT FP8 shipping student and non-CUDA backends are still unbuilt/untrained. Published held-out metrics remain v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-07-31):** headline reports `36,650 / 100,000`, loss `0.19268`, viz `36,000`, and ETA ~1.1d, but service health says **Trainer offline, stuck at step 36,650 for 60h**. Watcher is healthy (published 1s ago), CF Worker is HTTP 200 in 77ms, R2 is healthy, and GPU telemetry is unavailable. The identical step/loss and the prior brief's 36h counter establish that this is a real trainer/metrics outage; live/ETA copy is stale, not progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **[TSOG: A Format For Temporally And Spatially Ordered Gaussians](https://arxiv.org/abs/2607.28049)** (arXiv:2607.28049v1; Shady Gmira, Evangelos Alexiou, Emmanouil Potetsianakis, Emmanuel Thomas; 2026-07-30) — extends spatially ordered Gaussian attribute storage into the temporal domain, representing 4DGS attributes as index-aligned image data with a timeline parameterization; the abstract reports >90% size reduction with small PSNR changes versus PLY sequences and FreeTimeGS. **Why it matters to OSS:** the strongest new direct lead for #12: evaluate temporal ordering/delta layout and texture-friendly decode for persistent-canvas storage, bandwidth, serialization, and cross-platform GPU access. It is a representation/codec result, not an SR/FG runtime.
- **[GenSplatCodec: Feed-Forward Gaussian Splatting Compression via One-Step Diffusion](https://arxiv.org/abs/2607.24403)** (arXiv:2607.24403v1; Qiang Hu, Zhenlong Wu, Lei Huang, Zihan Zheng, Xiaoyun Zhang, Wenjun Zhang; 2026-07-27) — combines a compact structural stream, reference appearance stream, and geometry-guided one-step generative decoder for low-bitrate Gaussian splats. **Why it matters to OSS:** useful offline rate–distortion/teacher comparison for high-frequency temporal canvas content, but diffusion decode excludes it from the ≤2ms shipping path.

### Super-Resolution / Upscaling

- No newly verified direct real-time game SR, 2D-GS arbitrary-scale SR, or student-distillation paper cleared the relevance gate. Recent targeted results mainly re-found already-tracked NanoVSR and TRaM-VSR; keep NanoVSR as the #9/#4 reparameterized-student control.

### Frame Extrapolation / Generation

- No newly verified material single-network game frame-extrapolation paper. The dedicated Atom query was rate-limited; successful VSR/VFI and recent-category fallback checks did not expose a credible new OSS-FX baseline. Do not conflate codec interpolation or event-camera VFI with exact-engine-MV extrapolation.

### Inference Optimization

- TSOG is the actionable new layout lead. Prototype only after #13 provides baseline L2/DRAM data: compare temporally ordered texture/atlas records with the planned low-rank representation on bytes/Gaussian, update cost, L2 hit rate, DRAM traffic, raster time, and temporal/PSNR error.
- No new material TensorRT, INT8, or kernel-fusion SR paper cleared the shipping relevance gate.

## Archive.org Findings

- Advanced Search returned 3 noisy GitHub/archive snapshots for `"gaussian splatting" super-resolution`, mixed/irrelevant artifacts for `"frame generation" deep learning`, and no credible result for `"real-time upscaling" game`.
- Targeted acronym checks: GS-STVSR had 0 results; the sole GSASR hit was an unrelated 1961 CIA record; GRAPE and STSS were dominated by unrelated media. A narrowed Gaussian/splat/SR query returned no validated archival record.
- GSASR itself resolves to its live GitHub project. Targeted Wayback availability requests then returned HTTP 429, so there is **no confirmed archived/taken-down version**, not evidence that one does not exist.

## GitHub / Open Source

- **[SplatSuRe](https://github.com/pranav-asthana/SplatSuRe)** — official CVPR 2026 selective SR with multi-view-consistent 3DGS code; 86 stars, last pushed 2026-06-16, unasserted license. **High technical benchmark lead:** closest direct splat/SR competitor, but scope is multi-view 3D rather than OSS’s persistent screen-space temporal canvas.
- **[GSASR](https://github.com/ChrisDud0257/GSASR)** — ICCV 2025 arbitrary-scale 2D Gaussian-splat SR; 102 stars, Apache-2.0, last pushed 2025-08-06 (repo metadata updated 2026-07-14). **High baseline/reference value:** use for comparative tests and permissive implementation study; not a new push.
- **[RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** — RDNA3-oriented decoupled neural upscaler with G-buffer guidance; 15 stars, GPL-3.0, pushed 2026-07-24. **Cross-vendor data-contract signal:** inspect evaluation/input design only; GPL blocks code reuse.
- **[flux-fidelity](https://github.com/ajccarlson/flux-fidelity)** — 0-star Apache-2.0 WebGPU video upscale/interpolation project, pushed 2026-07-30. Watch its portable-WebGPU inference direction; it is too immature to treat as a competitor.
- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** — 9,570 stars, 343 forks, GPL-3.0, pushed 2026-07-31. **Positioning/integration threat:** dominant open bridge for DLSS2+/XeSS/FSR2+ and FSR-FG/XeFG. Study API input contracts, fallback/presentation behavior, and user expectations with strict GPL isolation.
- No credible newly surfaced open-source DLSS/FSR-analysis tool materially changes the landscape. Broad GitHub searches were spam-heavy and filtered.

## News

- **NVIDIA — primary verified:** [July 28 DLSS rollout](https://www.nvidia.com/en-us/geforce/news/halo-campaign-evolved-mistfall-hunter-corsair-cove-dlss/) says *Halo: Campaign Evolved* supports DLSS SR + FG and may be NVIDIA-app upgraded to DLSS 4.5 Dynamic MFG/latest SR; *Mistfall Hunter* supports SR, MFG, Dynamic MFG, Reflex and RT; *Corsair Cove* offers DLSS 4.5 SR, FG and up to 4× MFG, with NVIDIA-app Dynamic MFG up to 6× on RTX 50. **Impact:** OSS must model SR/FG/MFG/Dynamic-MFG, hardware generation, OS/API/backend, and game-native vs app-upgraded status as distinct capabilities—not a single vendor toggle.
- **Valve/AMD FSR 4.1 on Steam Machine/Proton — RSS lead only:** multiple July 29–31 press reports in [Google News RSS](https://news.google.com/rss/search?q=AMD%20FSR%204%20OR%20Radeon%20after%3A2026-07-16%20before%3A2026-08-01&hl=en-US&gl=US&ceid=US:en) claim Proton Experimental support. No accessible Valve/AMD primary release note was found; do not elevate this to a fact. If confirmed, it becomes a Linux/Proton FSR validation target.
- **AMD, Intel XeSS/Arc, Apple Metal/MetalFX, Unreal, Unity:** no qualifying new primary upscaling announcement was verified in the last 14 days.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or its metrics emitter. Require an increased step, fresh GPU telemetry, and new visualization before calling recovery; watcher/CDN health is insufficient.
- [ ] Deep-read TSOG and add a #12 microbenchmark for temporal ordering/texture-style packed records versus low-rank canvas layout; gate it behind #13 cache/DRAM profiling.
- [ ] Use GenSplatCodec only as an offline codec/teacher rate–distortion reference, never as a student-runtime candidate.
- [ ] Add SplatSuRe and GSASR to the research benchmark matrix: equal LR/HR, motion/G-buffer availability, latency, PSNR/LPIPS, temporal error, disocclusion behavior, and extrapolation quality.
- [ ] Build a capability matrix around OptiScaler and current DLSS deployment: temporal SR, FG, MFG/Dynamic-MFG, model/version, hardware gate, OS/API/backend, and native/app-injected availability.
- [ ] Await a Valve or AMD primary source before spending on SteamOS/Proton FSR 4.1 integration work.

## Bibliography Update

- Added TSOG and GenSplatCodec. TSOG is the priority #12 layout/decode paper; GenSplatCodec is retained as an offline compression reference.

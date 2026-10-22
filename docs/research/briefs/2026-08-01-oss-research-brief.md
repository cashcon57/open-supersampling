# OSS Research Brief — 2026-08-01

_Collection: local Git/README, GitHub REST, rendered public dashboard, arXiv Atom (six targeted queries; 34 deduplicated candidates), Archive.org Advanced Search, GitHub repository metadata, Apple/NVIDIA/Intel primary pages, and Google News RSS. RSS-only claims are explicitly labelled leads._

## Project Status Summary

- **Latest commits:** `a97a180` adds tailnet-gated training controls; `fcf8084` adds their API/dashboard panel. The preceding work preserves the v7 step-5K recovery snapshot, fixes cross-OS checkpoint resume and the WSL2 DataLoader deadlock, and adds FlashAttention 2/TF32/prefetch throughput work.
- **Open issues:** 12. Immediate technical gates remain #14 (v6.2 VRAM), #13 (Nsight L2 profile), #12 (packed low-rank canvas), #9 (reparameterized student), #8 (HIP/Metal/Vulkan/SYCL), #7 (FSR 4 teacher signal), and #4 (≤1M student distillation).
- **README snapshot:** `srcnn-v7.0-pico-005` is the HAT-Tiny research teacher: N-D `(x,y,t)` Gaussians, `V_xt` covariance motion, parent-child density, Mip-Splat filters, and alpha=0.5 OSS-FX supervision. The ≤1M TensorRT FP8 shipping student and non-CUDA backends remain unbuilt. Published held-out metrics remain v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-08-01):** headline says `36,650 / 100,000`, loss `0.19268`, viz `36,000`, and ETA ~1.1d, yet service health says **Trainer offline, stuck at step 36,650 for 84h**. Watcher is healthy (published 0s ago), CF Worker is HTTP 200 in 109ms, R2 origin is HTTP 200, and GPU telemetry is unavailable. The step/loss are unchanged from 2026-07-31 while the counter rose 60h→84h: this is a persistent trainer/metrics outage; the live/ETA copy is stale.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- No newly published **material** paper cleared the relevance gate. The targeted scan re-found [TSOG](https://arxiv.org/abs/2607.28049v1), [SplatStream v2](https://arxiv.org/abs/2607.25971v2), and [GrainGS v2](https://arxiv.org/abs/2607.21448v2), all already tracked in `BIBLIOGRAPHY.md`/recent briefs. TSOG remains the priority representation lead for #12: temporally ordered, index-aligned texture-like records could be benchmarked against the planned low-rank canvas layout.
- Recent S-Avatar, 4DHumanDiff, DynHair, medical-SR, and remote-sensing results were filtered as off-domain; they do not justify an OSS architecture decision.

### Super-Resolution / Upscaling

- No newly verified material real-time game SR, 2D-GS arbitrary-scale SR, or student-distillation paper. The results re-found already-tracked [NanoVSR](https://arxiv.org/abs/2607.10495v1), [TRaM-VSR](https://arxiv.org/abs/2607.22231v1), and [Efficient INT8 SISR](https://arxiv.org/abs/2604.20291v1). **Why this matters:** keep NanoVSR and Efficient INT8 SISR as #9/#4 controls—structural reparameterization plus deployment-aware quantization—not as new bibliography entries.

### Frame Extrapolation / Generation

- No newly verified material single-network/exact-engine-MV frame-extrapolation baseline. The recent hits were fluid-video diffusion, video coding, and event-video reconstruction; none transfers to OSS-FX’s sub-2ms, engine-motion-vector setting. Do not use them as architecture competitors.

### Inference Optimization

- No new material TensorRT/INT8/kernel-fusion paper beyond the tracked INT8 student result. The practical order remains: obtain #13 L2/DRAM measurements first, then compare TSOG-style temporal ordering/texture records with the planned packed low-rank layout on bytes/Gaussian, update cost, L2 hit rate, DRAM traffic, raster time, and temporal error.

## Archive.org Findings

- Advanced Search returned 3 GitHub/archive snapshots for `"gaussian splatting" super-resolution`, but no primary paper archive; `"frame generation" deep learning` returned mostly DLSS Swapper/FluidFrames builds plus an unrelated diffusion registration; `"real-time upscaling" game` returned 0.
- Targeted checks: GS-STVSR 0; GSASR’s only hit is an unrelated 1961 CIA document; `GRAPE` (16,548) and `STSS` (122) are acronym-noise. No validated archived/taken-down GS-STVSR, GSASR, GRAPE, or STSS version was found.

## GitHub / Open Source

- **No new credible direct OSS competitor or analysis tool.** The only fresh Gaussian-SR hit, [3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) (Apache-2.0, 4 stars, pushed 2026-07-28), is CT-specific and not a game/canvas baseline.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains a useful cross-vendor data-contract lead: RDNA3-oriented, decoupled G-buffer guidance, 15 stars, GPL-3.0, pushed 2026-07-24. Inspect inputs/evaluation only; do not reuse GPL code.
- **Integration/positioning watch:** [OptiScaler](https://github.com/optiscaler/OptiScaler) reached 9,601 stars / 346 forks and was pushed 2026-08-01. Its DLSS2+/XeSS/FSR2+ bridge plus FSR-FG/XeFG replacement behavior remains the user-facing compatibility benchmark. GPL-3.0 requires strict isolation.

## News

- **Apple — primary verified:** [WWDC26: Build real-time neural rendering pipelines with Metal](https://developer.apple.com/videos/play/wwdc2026/359/) documents three relevant Metal 4 routes: integrated MetalFX neural denoise/upscale, pre-trained models via the ML command encoder in the graphics command buffer, and tiny in-shader networks via TensorOps/cooperative tensors. It explicitly requires de-jittered motion vectors for MetalFX. **Opportunity:** make OSS’s future Metal backend a first-class scheduling/data-contract design, not a post-CUDA port; validate exact de-jittered MV semantics at the capture/shim boundary.
- **Intel — primary verified:** [Arc driver 32.0.101.8864](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html), dated 2026-07-20, supports Arc A/B and Core Ultra graphics. No XeSS feature change is claimed on the page; it is a validation-environment update, not a competitive technology release.
- **NVIDIA — primary verified:** [SIGGRAPH 2026](https://blogs.nvidia.com/blog/siggraph-news-2026/) highlights 3D-guided neural rendering, temporal stability, and real-time 4K as stated research objectives. It does **not** substantiate the RSS claims of a shipping “DLSS 5”; treat those headlines as unverified leads, not roadmap facts.
- **AMD/FSR 4.1 Steam Machine/Proton:** multiple RSS headlines claim support, but no Valve/AMD primary release note was located. Keep it as an unverified Linux/Proton test lead. No qualifying new primary Unreal/Unity upscaling announcement was verified.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or its metrics emitter. Acceptance requires a higher step, fresh GPU telemetry, and a new viz—not just a healthy watcher/CDN.
- [ ] Deep-read/prototype TSOG only after #13: benchmark temporally ordered texture records against #12 low-rank packing under OSS update/raster workloads.
- [ ] Make de-jittered motion-vector convention an explicit capture-tool/shim acceptance test before Metal work; investigate Metal 4 ML command encoder/TensorOps as the Apple student-runtime path.
- [ ] Keep NanoVSR/INT8 SISR as student controls for #9/#4; prioritize TensorRT-friendly fused/reparameterized graphs over transformer-like student designs.
- [ ] Track OptiScaler input/presentation behavior as the integration UX baseline; maintain GPL isolation.
- [ ] Obtain a Valve or AMD primary source before committing SteamOS/Proton FSR 4.1 integration effort.

## Bibliography Update

- No append: all material arXiv results from this scan were already represented in `BIBLIOGRAPHY.md`; avoiding duplicate entries.

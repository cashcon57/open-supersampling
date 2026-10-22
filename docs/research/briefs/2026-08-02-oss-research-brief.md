# OSS Research Brief — 2026-08-02

_Collection: local Git/README, GitHub REST, rendered public dashboard, arXiv Atom queries across the requested categories, Archive.org Advanced Search, GitHub repository metadata, AMD/NVIDIA/Intel primary pages, and Google News RSS. RSS-only claims are labelled accordingly._

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and their dashboard surface. The preceding commits preserve the v7 step-5K recovery snapshot, repair cross-OS checkpoint resume and the WSL2 DataLoader deadlock, and add FlashAttention 2 / TF32 / prefetch throughput work.
- **Open issues:** 12: #14 v6.2 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #9 reparameterized student, #8 HIP/Metal/Vulkan/SYCL ports, #7 FSR 4 teacher signal, #4 <=1M student distillation, plus v6.2 launch/kernel/ablation work.
- **README snapshot:** `srcnn-v7.0-pico-005` remains the HAT-Tiny research teacher: N-D `(x,y,t)` Gaussians with `V_xt` motion covariance, loss-adaptive density, Mip-Splat filters, and alpha=0.5 OSS-FX supervision. The <=1M TensorRT-FP8 shipping student and non-CUDA backends remain unbuilt. Published held-out quality is still v5-only: 25.703 dB PSNR / 0.1666 LPIPS on TartanAir `oldtown`.
- **Training dashboard (rendered 2026-08-02):** headline says `36,650 / 100,000`, loss `0.19268`, viz `36,000`, “live,” ETA ~1.1d; service health says **Trainer offline, stuck at step 36,650 for 23h**. Watcher is healthy (published 1s ago), CF Worker HTTP 200 in 95ms, R2 origin HTTP 200; GPU telemetry is unavailable. Step/loss are unchanged from the 2026-08-01 brief, while the stuck counter reset from 84h to 23h: this is a dashboard health-counter inconsistency, not recovery. The trainer/metrics emitter remains the operational blocker.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- No newly published material paper cleared the OSS relevance gate. The latest targeted hits re-surface **TSOG: A Format For Temporally And Spatially Ordered Gaussians** (Shady Gmira, Evangelos Alexiou, Emmanouil Potetsianakis, Emmanuel Thomas; 2026-07-30; [arXiv:2607.28049](https://arxiv.org/abs/2607.28049)), already tracked. Its temporally/spatially ordered, index-aligned image representation remains the strongest direct reference for #12 temporal delta organization, packed-canvas serialization, and texture/decode experiments.
- The other current 4D/temporal-Gaussian candidates were avatars, human reconstruction, robotics, or world models rather than real-time 2D reconstruction. They do not justify a v7 design change.

### Super-Resolution / Upscaling

- No new material real-time game SR, 2D-GS arbitrary-scale SR, or student-distillation paper. The scan re-found already-tracked **NanoVSR** ([arXiv:2607.10495](https://arxiv.org/abs/2607.10495)) and **TRaM-VSR** ([arXiv:2607.22231](https://arxiv.org/abs/2607.22231)). Keep NanoVSR’s structurally reparameterized convolutional deployment approach as a #9/#4 student control, not a new bibliography item.

### Frame Extrapolation / Generation

- No newly verified material single-network or exact-engine-motion-vector extrapolation baseline. Recent results were diffusion video, coding, fluid generation, or event-video reconstruction—useful neither as a <=2ms shipping design nor a direct OSS-FX competitor.

### Inference Optimization

- No newly material TensorRT, INT8-SR, or kernel-fusion paper. The evidence still prioritizes #13 profiling before selecting a packed layout: compare TSOG-like ordered records and the proposed low-rank layout on bytes/Gaussian, L2 hit rate, DRAM traffic, raster time, and temporal error.

## Archive.org Findings

- Advanced Search returned three archived GitHub/star-list snapshots for `"gaussian splatting" super-resolution`; `"frame generation" deep learning` yielded FluidFrames/DLSS Swapper builds and an unrelated diffusion record; `"real-time upscaling" game` yielded 0.
- Known-work checks remain noisy: GS-STVSR 0, GSASR one unrelated 1961 CIA item, GRAPE 16,537 unrelated results, STSS 122 unrelated results. No validated archived, taken-down, or updated version of GS-STVSR, GSASR, GRAPE, or STSS was found. This is not evidence that no archival copy exists.

## GitHub / Open Source

- **No new credible direct OSS competitor or DLSS/FSR analysis tool.** The only fresh Gaussian-SR result, [3DCT_SR_DDNM_NAB-GS](https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS) (Apache-2.0, 4 stars, pushed 2026-07-28), is CT-specific.
- [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) remains a useful data-contract lead: RDNA3-focused neural upscaler with decoupled G-buffer guidance, 15 stars, pushed 2026-07-24, GPL-3.0. Inspect input/evaluation semantics only; do not reuse GPL code.
- **Integration/positioning watch:** [OptiScaler](https://github.com/optiscaler/OptiScaler) has 9.6k stars / 346 forks and a 2026-07-31 commit adding a DLSSG resource-tagging workaround using DLSS inputs. Its DLSS2+/XeSS/FSR2+ bridge and FSR-FG/XeFG replacement behavior remains the compatibility UX benchmark. GPL-3.0 requires strict isolation.

## News

- **AMD — primary verified, high competitive significance:** GPUOpen’s June 24 [FSR SDK 2.3 update](https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/) brings ML FSR Upscaling 4.1.1 from RDNA4 to RX 7000/RDNA3, FSR Frame Generation 4.0.1 with improved motion-vector/camera-parameter handling, and Ray Regeneration 1.2. AMD says analytical SR/FG fallbacks cover RDNA3.5/RDNA2 and older hardware. **Threat:** FSR’s single SDK is closing hardware-coverage gaps while retaining separate ML SR/FG paths. **Opportunity:** its explicit MV/camera corner cases reinforce OSS’s exact engine-MV contract as a testable differentiator.
- **NVIDIA — primary verified:** [SIGGRAPH 2026](https://blogs.nvidia.com/blog/siggraph-news-2026/) presents 3D-guided neural rendering, temporal stability, and real-time 4K as research goals. It does not announce a shipping “DLSS 5”; RSS headlines claiming that remain unverified and should not inform OSS roadmap decisions.
- **Intel — primary verified:** Intel’s [Arc driver 32.0.101.8864](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html), dated 2026-07-20, supports Arc A/B and Core Ultra hardware. The official page states no XeSS feature change: validation-environment update, not a competitor feature release. RSS-only XeSS SDK 3.0.2 claims were not promoted without an official SDK source.
- **Apple / engines:** no qualifying new primary Apple Metal, UE5, or Unity upscaling announcement was found in the monitoring window. Keep the prior Metal 4 motion-vector/ML-command-encoder follow-up active.
- **FSR 4.1 / Proton:** RSS reports Valve/Steam Machine/Proton support, but no authoritative Valve announcement was retrieved. Treat as an unverified Linux test lead—not a confirmed platform commitment.

## Action Items / Follow-ups

- [ ] **Critical:** restore/restart the 3080 Ti trainer or metrics emitter. Require a higher step, fresh GPU telemetry, and a new viz; healthy watcher/CDN alone is insufficient.
- [ ] Deep-read/prototype TSOG only after #13: benchmark ordered texture records against #12 low-rank packing under OSS update/raster workloads.
- [ ] Add explicit capture-tool/shim tests for de-jittered motion-vector and camera-parameter conventions; use AMD’s FG 4.0.1 corner-case emphasis as motivation.
- [ ] Keep NanoVSR/INT8 SR as #9/#4 student controls; favor TensorRT-friendly fused/reparameterized CNN graphs over transformer-like student designs.
- [ ] Track OptiScaler’s DLSS resource-tagging/input behavior as the DLL-shim integration UX baseline, with GPL isolation.
- [ ] Obtain a Valve primary source before committing SteamOS/Proton FSR 4.1 work.

## Bibliography Update

- No append: no newly material verified arXiv paper was found; all relevant re-discoveries are already represented in `BIBLIOGRAPHY.md`.

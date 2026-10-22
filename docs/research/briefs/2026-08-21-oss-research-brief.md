# OSS Research Brief — 2026-08-21

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`.
- **Open issues:** 12. The critical implementation chain remains **#12 packed low-rank canvas → #13 Nsight L2 profile → #14 v6.2 VRAM measurement**; #1 rasterizer, #3 tile bin, #4/#7/#9 distillation, #8 cross-vendor kernels, and #10 G-buffer ablation remain open.[1]
- **README snapshot:** v7 remains the active research-teacher design: native `(x,y,t)` Gaussian primitives with `V_xt` covariance coupling, loss-adaptive density, and time-slice OSS-FX. The ≤1M shipping student, DLL shim, and non-CUDA kernels are still unbuilt.
- **Training dashboard:** `srcnn-v7.0-pico-005` remains at **36,650** with latest total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263; 2,304 primitives). Fresh dashboard JSON reports the trainer **offline/stuck for 290h**, while watcher, CF Worker, and R2 are healthy.[2][3] This is a stale-active dashboard contradiction, not training progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models](https://arxiv.org/abs/2608.19556)** (arXiv:2608.19556; Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh; 2026-08-20) — uses 4DGS-based consistency signals to reduce geometric drift in long autoregressive diffusion-video rollouts. **Why it matters to OSS:** a useful failure taxonomy for time-axis state drift, but diffusion-scale inference makes it unsuitable for the shipping path; no architecture change recommended.[4]
- No newly posted direct 2D Gaussian arbitrary-scale SR / GS-STVSR / GRAPE competitor survived the relevance gate. Reconfirmed DAV4’s manifold-safe 4DGS parameter updates as a teacher-side covariance/opacity-stability reference only; it is not a real-time candidate.[5]

### Super-Resolution / Upscaling
- The targeted metadata scan surfaced no newly material student/deployment paper beyond already-tracked NanoVSR, MeanSR, MotionCraft, and quantized-SR controls. Current results were either UHD/face/medical restoration or too expensive/underspecified for a ≤1M, <2 ms game student.

### Frame Extrapolation / Generation
- No newly material single-network frame-extrapolation result displaced existing ANVIL, Amulet, 3DGS³, or SNM-VFI controls. Recent interpolation candidates rely on diffusion, event-camera inputs, or learned-flow ambiguity resolution; retain them as invalid-MV/hole-filling research controls, not OSS-FX deployment designs.

### Inference Optimization
- No fresh TensorRT/INT8/kernel-fusion paper met the deployment relevance gate. The prior RoofGS and QuARC-GS recommendations stand: choose packing, key width, quantization, and fusion boundaries only after #12–#14 measure OSS’s actual bandwidth, L2 locality, occupancy, and quality trade-offs.

## Archive.org Findings
- The requested Advanced Search inventories remain noisy: 3 Gaussian-SR records are archived GitHub lists; `frame generation deep learning` (225 hits) and `real-time upscaling game` (107 hits) are dominated by unrelated media/software records. No preserved primary GS-STVSR, GRAPE, or STSS artifact surfaced.[6]
- Targeted Wayback checks confirmed captures for the GSASR repository (2025) and GRAPE’s arXiv page (2025); the GS-STVSR arXiv URL returned no captures in this query. Archive absence is not evidence the paper was removed.[7]

## GitHub / Open Source
- GitHub search over the last 14 days found no material new Gaussian-SR, open frame-generation, neural-upscaler, or DLSS/FSR-analysis implementation.[8]
- **[MAKO](https://github.com/eugeniosegala/MAKO)** is the only practical integration lead: a **168-star**, actively updated Vulkan/Linux layer for Lossless Scaling frame generation on Deck/Steam Machine/desktop Linux. **Opportunity:** inspect presentation recovery, per-game configuration, and Vulkan integration boundaries. **Not a competitor:** it depends on proprietary `Lossless.dll` and has no asserted SPDX license.[9]
- `gomzee1024/SelectiveSR` is active but has 0 stars and no license; its claimed Gaussian-SR path lacks a temporal/game/runtime story. Monitor only.

## News
- **Apple (primary):** MetalFX’s redesigned temporal upscaler uses the Neural Engine and M5 Pro/Max Neural Accelerators, accepts dynamic-resolution subrectangles and motion vectors, and supports distortion fields.[10] **Opportunity:** make those input conventions explicit in the future Metal ABI now.
- The 14-day Google News watch contains FSR 4.1 / DLSS 4.5 game-support reports (including Alan Wake 2 and Beast of Reincarnation), but these are RSS/third-party leads rather than new primary vendor architecture announcements; do not treat them as competitive capability changes until vendor/game release notes are verified.[11]
- No material primary NVIDIA, AMD, Intel Arc/XeSS, GPU-announcement, Unreal, or Unity update was found beyond the already-tracked FSR 4.1, XeSS 3, DLSS 4.5, and MetalFX positioning.

## Action Items / Follow-ups
- [ ] **Critical:** restart/restore `srcnn-v7.0-pico-005` or correct its dashboard state; it has not advanced past 36,650 and reports **290h** offline.
- [ ] Execute #12 → #13 → #14 before committing to a packed-attribute format, quantization strategy, or fused raster kernel.
- [ ] Deep-read Stream4D only for long-horizon time-axis consistency failure modes; reject any diffusion or scene-scale inference dependency.
- [ ] Specify the Metal input ABI: dynamic-resolution subrectangles, exact motion-vector convention, distortion/postprocess masks, and quantized-weight fallback.
- [ ] Verify RSS-reported FSR 4.1 / DLSS 4.5 game integrations from primary release notes before changing competitive claims.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://opensupersampling.com/data.json
[3] https://opensupersampling.com/status.json
[4] https://arxiv.org/abs/2608.19556
[5] https://arxiv.org/abs/2608.18388
[6] https://archive.org/advancedsearch.php
[7] https://web.archive.org/cdx?url=https%3A%2F%2Fgithub.com%2FChrisDud0257%2FGSASR&output=json&limit=3&filter=statuscode%3A200&collapse=digest
[8] https://api.github.com/search/repositories
[9] https://api.github.com/repos/eugeniosegala/MAKO
[10] https://developer.apple.com/metal/whats-new
[11] https://news.google.com/rss/search?q=FSR+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen

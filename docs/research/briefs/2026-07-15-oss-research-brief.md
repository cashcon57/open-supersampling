# OSS Research Brief — 2026-07-15

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
- Open issues: #14 D3 VRAM footprint; #13 Nsight L2 hit-rate profile; #12 packed low-rank canvas layout; #10 decoupled G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR student-backbone ablation; #8 cross-vendor kernel ports; #7 FSR 4 binary teacher-signal distillation; #6 v6.2-pico-002 launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: pre-alpha; current architecture is v7 `srcnn-v7.0-pico-005`, an N-D Gaussian canvas with native time axis, HAT-Tiny teacher, parent-child loss-adaptive density, Mip-Splatting filters, and alpha=0.5 OSS-FX supervision. Shipping ≤1M-param student is still not trained.
- Training dashboard: rendered page shows `srcnn-v7.0-pico-005` at step 35,200 / 100,000, loss 0.10166, viz @ step 35,077, headline says "updated just now" / "live". Service health contradicts that: **Trainer — offline, stuck at step 35,200 for 46h**; Watcher healthy published 0s ago; CF Worker healthy HTTP 200 in 106ms; R2 origin healthy HTTP 200; GPU stats unavailable / host offline or rsync failed. The step is unchanged while loss/viz metadata moved, so the dashboard state remains internally inconsistent and operationally stale.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements** (arXiv:2607.12362v1, 2026-07-14; Seung-gyeom Kim, Areum Kim, Yongjae Yoo, Sukmin Yun) — Introduces SPIN-4DGS, predicting Gaussian attributes from explicit spatiotemporal positions with a lightweight feed-forward network to handle fast motion and large inter-frame displacement. Why it matters to OSS: directly relevant to v7 native-time Gaussian attributes and the failure mode OSS-FX will hit on fast camera/object motion.
- **TemporalGS: Training-Free Plug-and-Play Acceleration for 3D Gaussian Splatting Rendering via Temporal Priors** (arXiv:2607.03390v1, 2026-07-03; Yuhongze Zhou, Zihao Yang, Xinxin Zuo, Juwei Lu) — Uses temporal priors to reduce redundant preprocessing, sorting, and rasterization across consecutive 3DGS frames. Why it matters to OSS: already tracked in bibliography; concrete cache/reuse idea for active-mask pruning and tile/raster work.
- **MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting** (arXiv:2607.01578v1, 2026-07-02; Jianwei Hu et al.) — Adds per-Gaussian motion variance and temporal attention for dynamic/static separation. Why it matters to OSS: already tracked; compare against v7 `V_xt` cross-correlation and parent-child density logic.
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v2, 2026-06-28; Giulio Federico et al.) — Replaces low-pass Gaussian color with Gaussian-Hermite "Brane" modes for local high-frequency detail. Why it matters to OSS: already tracked; candidate primitive ablation if vanilla Gaussians blur edges/textures.
- **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.29400v1, 2026-06-28; Giulio Federico et al.) — Feed-forward adaptive Gaussian allocation for ASR. Why it matters to OSS: already tracked; maps onto v7 parent-child density and packed-canvas budgeting.

### Super-Resolution / Upscaling
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495v1, 2026-07-11; Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies) — Fully convolutional VSR with structural reparameterization into standard inference convolutions, avoiding explicit flow/custom ops while preserving temporal alignment through progressive training. Why it matters to OSS: strong candidate reference for the ≤1M-param student, especially RepVGG/RepSR-style ablation issue #9 and TensorRT-friendly deployment.
- **Real-Time Neural-Enhancement for Online Cloud Gaming** (arXiv:2501.06880v1, 2025-01-12; Shan Jiang et al.) — Cloud-gaming SR framework reuses fine-tuned model variants to reduce per-stream adaptation latency. Why it matters to OSS: not a runtime architecture match, but useful competitive/context signal for game/cloud delivery and repetitive-scene adaptation.
- **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699v2, 2023-08-13; Jia Li et al.) — Separates lighting/material components to reduce ghosting and preserve texture in real-time rendering SR. Why it matters to OSS: useful G-buffer/material-channel reference for texture/detail recovery without hallucinating inconsistent history.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1, 2023-08-03; Antoine Mercier et al.) — Gaming neural supersampling with motion vectors/depth and rendering-specific data. Why it matters to OSS: baseline/dataset precedent for engine-input claims.

### Frame Extrapolation / Generation
- No newly surfaced primary arXiv paper from the narrow frame-extrapolation queries beat the already-tracked Gaussian/time-axis leads today. Best adjacent leads remain `Implicit 4D Gaussian Splatting` for fast motion and `MVFusion-GS` / `TemporalGS` for temporal Gaussian state handling.
- Watchlist only: Google News/RSS surfaced headline-level FSR 4 multi-frame-generation rumors; no primary AMD source verified in this run.

### Inference Optimization
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (arXiv:2607.10495v1) — Hardware-compatible structural reparameterization is the highest-value optimization paper this run. Why it matters to OSS: likely maps directly to the planned tiny student and the open RepVGG/ECBSR/RepSR ablation.
- **QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms** (arXiv:2303.04336v2) — Older but relevant low-latency mobile SR architecture. Why it matters to OSS: useful sanity check for Pico/handheld latency targets, but single-image only.

### Game Engine Integration
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483v1) — Gaming SR dataset/method with rendering auxiliary buffers. Why it matters to OSS: still one of the cleaner public references for motion-vector/depth-conditioned game supersampling evaluation.
- **Real-Time Neural-Enhancement for Online Cloud Gaming** (arXiv:2501.06880v1) — Cloud-game SR adaptation/reuse system. Why it matters to OSS: relevant to hosted demos/streaming deployment more than local DLL-shim runtime.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 item hits, mostly broad GitHub/star-list archives rather than primary paper artifacts.
- `"frame generation" "deep learning"`: 3 hits: DLSS Swapper Windows builds, FluidFrames Windows builds, and an OSF record for MoCha-LD; useful as ecosystem archive leads, not OSS architecture inputs.
- `"real-time upscaling" game`: no useful Archive.org item metadata returned.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,636 hits but acronym pollution dominates (especially `GRAPE`); not useful without targeted URL checks.
- Targeted Wayback CDX: `arXiv:2605.11489` / 3DGS3 has a 2026-06-07 HTTP 200 capture; `github.com/ChrisDud0257/GSASR` has HTTP 200 captures from 2025-03-13 and 2025-08-01; `arXiv:2604.18047` / GS-STVSR returned no 200 rows in the limited check; SplatSuRe and GS-STVSR GitHub CDX checks timed out.

## GitHub / Open Source
- `gaussian splatting super-resolution` (`pushed:>=2026-07-01`): 1 result, `GiulioFede/ResonantBraneSplatting` (0★, license not declared, pushed 2026-07-01) — implementation lead for RBS; verify code maturity before using as a baseline.
- `frame generation open source`: 1 result, `OULDRABAH95/AllPen-framework-` — unrelated security/agent framework; no OSS action.
- `neural upscaler`: 7 results; most are toy/site repos. Notable overlap: `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling` (4★, MIT, pushed 2026-07-02) — static structural analysis of AMD FSR 4.1.0 neural upscaler with DXIL entrypoint inventory, weight extraction, blob layout, and rebuild verification. Why it matters: possible external reverse-engineering signal for competitor architecture, but treat as unverified until code is inspected.
- `DLSS FSR analysis`: no recently pushed repositories returned.
- `FSR 4 upscaler`: same `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling` result.
- `XeSS upscaler open source`: no recently pushed repositories returned.
- Vendor/direct-page probes: NVIDIA GeForce news page fetched and contains DLSS hits; AMD GPUOpen news fetched and contains FSR/FidelityFX/upscaling/super-resolution/frame-generation hits; AMD FidelityFX SDK URL returned 404; Intel XeSS and Unreal feed returned 403; Unity blog fetched with no relevant keyword hits.

## News
- **NVIDIA / DLSS 4** (Google News RSS, headline-level): NVIDIA item says `DOOM: The Dark Ages | Revelations` launched with DLSS Multi Frame Generation and path tracing on 2026-07-07; Wccftech headline says `Gears of War: E-Day` will launch with "DLSS 4.5 Dynamic Multi Frame Generation" and ray tracing. Treat non-NVIDIA headlines as leads until primary pages are verified.
- **AMD / FSR 4** (Google News RSS, headline-level): Digital Foundry/TechSpot coverage around FSR 4.1 on RDNA 3; TechPowerUp headline claims AMD FSR multi-frame generation up to 8x is in development; TweakTown headline claims experimental FSR 4/4.1.1 builds running on older GPUs via OptiScaler/Proton. Threat/opportunity: AMD appears to be pushing both wider hardware coverage and multi-frame-generation narrative; OSS should keep emphasizing one-architecture SR+FG and vendor-neutrality.
- **Intel Arc / XeSS**: no strong primary XeSS update found; RSS mostly benchmarks/noisy hardware articles.
- **Apple Silicon / Metal**: RSS showed general Apple/Metal gaming and M4 GPU benchmark items, but no primary upscaler/runtime update verified.
- **Unreal / Unity**: no useful recent RSS items from the query; Unreal feed direct fetch was 403, Unity blog had no relevant keyword hits.

## Action Items / Follow-ups
- [ ] Ops: trainer is still reported offline/stuck at step 35,200, now for 46h while dashboard headline remains live; verify trainer process on the 3080 Ti host and fix stale/contradictory health semantics.
- [ ] Deep-read **Implicit 4D Gaussian Splatting / SPIN-4DGS** for fast-motion handling; map its explicit spatiotemporal-position attribute prediction onto v7 `t` axis and `V_xt` covariance.
- [ ] Deep-read **NanoVSR** and translate structural reparameterization into the open RepVGG/ECBSR/RepSR student-backbone ablation (#9).
- [ ] Inspect `Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling` before citing it; if real, extract concrete competitor architecture/weight-layout facts for OSS positioning.
- [ ] Re-run targeted Wayback CDX for GS-STVSR/GitHub when archive.org is responsive; today’s targeted checks timed out for some URLs.

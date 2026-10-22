# OSS Research Brief — 2026-07-25

_Collection: local repository/README, GitHub REST, rendered training dashboard, arXiv Atom/direct-metadata fallback, Archive.org Advanced Search, GitHub repository search, Google News RSS, and previously verified primary vendor pages. ArXiv Atom returned HTTP 429 for all three targeted queries; do not read the lack of fresh Atom results as no new research._

## Project Status Summary
- **Latest commits:** `a97a180` adds tailnet-gated dashboard controls; `fcf8084` adds the training-control API/panel. The preceding recovery and throughput work preserves the v7 Pico checkpoint, keeps `num_workers=0` for WSL2, enables cross-OS resume/FlashAttention-2/TF32 prefetch, and retains raster smoothing in FP32 under bf16.
- **Open issues:** **12** open. Highest-impact near-term items: #14 v6.2 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #9 reparameterized student backbone, #8 cross-vendor ports, #7 FSR 4 teacher signal, and #4 ≤1M student distillation.
- **README snapshot:** v7 `srcnn-v7.0-pico-005` remains a HAT-Tiny research teacher with native `(x,y,t)` Gaussians and `V_xt` covariance motion. The ≤1M-param TensorRT-FP8 student and all non-CUDA runtimes remain unbuilt. Published validation remains v5 only: 25.703 PSNR / 0.1666 LPIPS on held-out TartanAir `oldtown`.
- **Training dashboard:** hero reports **35,700 / 100,000**, loss **0.17966**, viz **35,077**, ETA ~1.1d. Health separately says **Trainer offline, stuck at 35,700 for 28m**; watcher is healthy (published 1s ago), CF Worker HTTP 200/89ms, R2 HTTP 200, and GPU stats are unavailable. Compared with the earlier same-day 35,250/42h report, training advanced 450 steps and the stuck counter reset; this is a fresh stall, not proof that host telemetry is reliable.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Future Rendering ≠ Future Surface: A Benchmark and Dataset for Dynamic Surface Reconstruction Beyond the Observed Window** — Yukun Shi, Minglun Gong; **2026-07-23**; [arXiv:2607.21471](https://arxiv.org/abs/2607.21471). FutureSurf withholds a sequence's final 25% and distinguishes future geometry from visually plausible future rendering; dynamic-Gaussian baselines retain a 2.0–6.6× future/observed error gap. **Why it matters to OSS:** use it as the direct OSS-FX evaluation template—score observed-prefix → held-out-future degradation plus motion/occlusion correctness, not rendered-frame quality alone.
- **GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis** — Jiahao He et al.; **2026-07-23**; [arXiv:2607.21448](https://arxiv.org/abs/2607.21448). Hierarchical anchors, temporal offsets, canonical/residual appearance, and stop-gradient isolation protect canonical positions from deformation updates. **Why it matters to OSS:** already tracked on 2026-07-24; it is a direct ablation reference for stable v7 persistent state versus `V_xt`/temporal updates and loss-adaptive density.

### Super-Resolution / Upscaling
- No fresh directly material game/VSR result was primary-verified in this run. **NanoVSR** ([arXiv:2607.10495](https://arxiv.org/abs/2607.10495), previously tracked) remains the appropriate reparameterized-CNN control for #9/#4.

### Frame Extrapolation / Generation
- No new direct frame-extrapolation architecture passed the relevance gate beyond FutureSurf. Its strategic warning applies directly: visually credible extrapolation can still contain wrong surface/motion state.

### Inference Optimization
- **Finding Fast Filters** — Karima Ma, Andrew Adams, Jonathan Ragan-Kelley; **2026-07-22**; [arXiv:2607.20634](https://arxiv.org/abs/2607.20634). Searches multirate, recurrent, summed, and cascaded FIR approximations, then lowers selected filters to fused vectorized parallel code. **Why it matters to OSS:** a methodology for EWA/Mip-splat smoothing/composite-filter approximations before manual per-vendor kernels; gate on shimmer and PSNR, not throughput alone.
- **Rendering 3D Gaussians on a Graph Processor** — Nicholas Fry, Ignacio Alzugaray, Mark Pupilli, Paul H. J. Kelly, Andrew J. Davison; **2026-07-17**; [arXiv:2607.15951](https://arxiv.org/abs/2607.15951). An SRAM-only tiled 3DGS renderer identifies routing bandwidth, per-tile capacity, and density imbalance as core bottlenecks. **Why it matters to OSS:** independently reinforces #12/#13: obtain L2/DRAM/tile-density traces before choosing the packed canvas layout.

## Archive.org Findings
- Advanced Search remains noisy: `gaussian splatting super-resolution` returned three archived GitHub-star-list snapshots; `frame generation deep learning` returned 213 mostly unrelated items; `real-time upscaling game` returned 106 unrelated archive items. `GS-STVSR`, `GRAPE gaussian splatting`, and `STSS super-resolution` returned zero; `GSASR` returned an unrelated 1961 record.
- No recoverable archival paper copy, takedown, or update was verified. These broad-search results are not evidence that the named papers were never archived or were removed.

## GitHub / Open Source
- Fresh GitHub REST searches (`pushed:>=2026-07-11`) found no credible new Gaussian-SR or DLSS/FSR-analysis repository. The frame-generation query was dominated by unrelated projects and an OptiScaler fork, not a new upstream signal.
- **RDNU** ([Adam-AtlasSoftware/RDNU](https://github.com/Adam-AtlasSoftware/RDNU)) remains the closest credible open G-buffer-guided neural-upscaler baseline: 15 stars, pushed 2026-07-24, RDNA3-focused, trained/fine-tuned on GameIR, vKITTI2, and Sintel. **Competitive note:** GPL-3.0 blocks direct reuse in Apache OSS, but its data contract and evaluation are worth monitoring.
- No new credible open-source DLSS/FSR analysis tool was verified.

## News
- **NVIDIA / DLSS:** previously primary-verified DLSS 4.5 material remains the relevant competitive move: second-generation transformer SR and Dynamic/6× Multi Frame Generation, with UE plugin package 8.7.2 dated 2026-07-21. **Threat:** OSS needs latency and artifact evidence for lower-marginal-cost extrapolation, not only a one-architecture claim.
- **AMD / FSR:** AMD GPUOpen's UE FSR 4.0.3a package supports UE 5.2–5.7, ML SR/FG on RX 9000+ and analytic fallbacks below; its WPO-motion-vector warning is a useful OSS capture/integration acceptance test for velocity coverage and ghosting.
- **Intel / Arc:** recent RSS/official-search signals point to Arc/XeSS Multi-Frame Generation messaging, but no new concrete XeSS SDK or engine integration was primary-verified in this run.
- **Apple / Metal and engines:** no material primary Apple MetalFX, Unreal, or Unity upscaling update was verified in the past 14 days.

## Action Items / Follow-ups
- [ ] **CRITICAL:** inspect the 3080 Ti trainer and metrics producer. The run advanced but is again offline at 35,700; watcher/CDN health only proves publication.
- [ ] Deep-read **FutureSurf**; add an OSS-FX held-out-future slice with future/observed degradation, engine-MV consistency, and disocclusion metrics.
- [ ] Combine #13 and #12: collect Nsight L2/DRAM/tile-density traces before committing to low-rank packed canvas layout; compare local-tile ownership and active-mask variants.
- [ ] Prototype automatic fast-filter search only if EWA/Mip-splat filtering materially costs the student path; reject candidates that regress temporal shimmer or PSNR.
- [ ] Monitor RDNU and open frame-generation middleware; run a shared Sintel/TartanAir quality-plus-latency comparison if reproducible weights/runtime ship.
- [ ] Competitive watch: DLSS 4.5 Dynamic MFG and AMD ML SR+FG raise the burden of proof for OSS's unified SR/FX positioning.

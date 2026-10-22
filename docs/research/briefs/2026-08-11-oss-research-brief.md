# OSS Research Brief — 2026-08-11

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`; local clone was missing at start of this run and was re-cloned from `origin/main` (clean, shallow `main`). The remote's last push remains 2026-06-08, so there is no new code activity in the last ~64 days.
- **Open issues:** 12. The immediate architecture/perf cluster is #12 packed low-rank canvas layout, #13 Nsight L2 profile, #14 measured v6.2 VRAM; shipping-model work remains #4 HAT-Tiny student, #7 FSR-4-teacher distillation, #8 cross-vendor ports, #9 reparameterized student ablation, and #10 G-buffer encoder ablation.
- **README snapshot:** v7 is an N-D `(x,y,t)` Gaussian canvas with 3D Cholesky `V_xt` motion correlation; the planned ≤1M CNN student has not been trained.
- **Training dashboard:** `srcnn-v7.0-pico-005` is at step **36,650/100,000**, loss **0.19268**, with a displayed ETA of ~1.1 days. Trainer is explicitly **offline/stuck for 51h** while the watcher, CF Worker, and R2 origin are healthy. GPU stats are unavailable. Dashboard copy still labels the run "live" despite the offline health state; treat this as an operational contradiction, not progress. Best published OSS result remains v5 pixel-temporal at 25.70 dB / 0.1666 LPIPS on held-out TartanAir oldtown.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting](https://arxiv.org/abs/2604.18047)** (Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20; arXiv:2604.18047; **previously tracked**) — A 2D-Gaussian continuous space-time VSR system that uses flow-guided center/color evolution, covariance resampling alignment to avoid covariance drift, and an adaptive motion-offset window; it reports near-constant X2–X8 temporal inference cost and >3× speedup at X32. **Why it matters to OSS:** direct external validation of covariance resampling and a single continuous SR+intermediate-frame representation; deep-read the drift-control and large-motion mechanisms against v7's `V_xt` covariance formulation.
- **[TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting](https://arxiv.org/abs/2604.00538)** (Suwoong Yeom, Joonsik Nam, Seunggyu Choi, Lucas Yunkyu Lee, Sangmin Kim, Jaesik Park, Joonsoo Kim, Kugjin Yun, Kyeongbo Kong, Sukju Kang; 2026-04-01; arXiv:2604.00538) — Continuous SE(3) transforms, hierarchical Bézier residuals, and local anchors preserve Gaussian identity over 600–1200-frame dynamic sequences instead of repeatedly respawning primitives. **Why it matters to OSS:** a concrete mitigation for v7 canvas identity/memory growth; evaluate a 2D analogue for long-lived rigid regions before increasing parent-child density.
- **[STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation](https://arxiv.org/abs/2604.10910)** (Jierun Lin, Jiacong Chen, Qingyu Mao, Shuai Liu, Xiandong Meng, Fanyang Meng, Yongsheng Liang; 2026-04-13, revised 2026-04-14; arXiv:2604.10910) — Separates static spatial and dynamic temporal hash encodings with key-frame canonical initialization, reporting +0.98 PSNR over compared Gaussian video representations. **Why it matters to OSS:** supports explicitly separating stable canvas state from temporal/motion state; a compact hash-conditioned residual could reduce motion-feature bandwidth in the student.

### Super-Resolution / Upscaling
- No newly published, directly game-relevant student-distillation paper was verified in this sweep. Broad TensorRT/INT8 searches were dominated by unrelated deployment papers; do not treat them as evidence for OSS's <2 ms target.

### Frame Extrapolation / Generation
- **[VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos](https://arxiv.org/abs/2605.10567)** (Nengbo Lu, Bin Zhao; 2026-05-11; arXiv:2605.10567; ICME 2026 accepted) — Learns per-Gaussian velocity with a physics code, particle-dynamics system, and global physical constraints, reporting strong novel-view interpolation and future-frame extrapolation. **Why it matters to OSS:** the engine supplies better motion observability than the paper's video-only setting, but its velocity-consistency objectives could regularize `V_xt`/alpha supervision and expose extrapolation failures at disocclusions.

### Inference Optimization
- No material paper addition. The actionable result is negative: current literature-search hits did not supply a tested replacement for OSS's planned packed low-rank layout, custom tile bin, and fused FP8/INT8 student path. Issue #12/#13 measurements remain the right prerequisite.

## Archive.org Findings
- Archive.org advanced search produced only noisy captures: three GitHub/archive snapshots for `"gaussian splatting" super-resolution`, and no result for `"real-time upscaling" game`.
- `"frame generation" deep learning` returned FluidFrames/DLSS Swapper builds plus an unrelated diffusion-registration record; no primary GS-STVSR, GSASR, GRAPE, or STSS archival copy was verified.
- A targeted Wayback CDX check for GS-STVSR's arXiv page timed out (HTTP 504). This is an archive-service failure, not evidence of absence; the live arXiv abstract is available.

## GitHub / Open Source
- **[optiscaler-client](https://github.com/interceptballast918/optiscaler-client)** — a small MIT C# Windows client (3 stars, pushed 2026-08-11) advertising AI upscaling and frame generation via the OptiScaler ecosystem. **Positioning signal:** continued demand for cross-vendor/drop-in upscaler switching, but it is integration tooling rather than a competing reconstruction model.
- **[bananascaler](https://github.com/julesklord/bananascaler)** — new MIT Go WIP GPU video-upscaler TUI (created 2026-07-16; pushed 2026-08-04; 0 stars). Not a game/runtime competitor; monitor only if it produces reusable low-latency deployment plumbing.
- GitHub search found no newly active open game-focused Gaussian-SR implementation. The sole recent Gaussian-SR hit is a medical 3D CT project and is not relevant to OSS architecture.
- No credible new open-source DLSS/FSR analysis tool was found. NVIDIA's Streamline remains the integration reference for multi-IHV plugin plumbing.

## News

### NVIDIA / DLSS — threat: moving quality and FG ceiling
- NVIDIA's current developer page says **DLSS 4.5** adds a second-generation transformer SR model and Dynamic Multi Frame Generation; it advertises up to five generated frames per rendered frame and UE 5.5–5.8 plugin package 8.7.2 dated 2026-07-21. **Why it matters:** OSS must benchmark both current-frame quality and temporal failure modes against a transformer teacher, while preserving its one-canvas cost advantage rather than claiming feature parity on generated-frame multiplier.

### AMD / FSR — opportunity and threat
- AMD's FSR SDK **2.3** exposes ML FSR Upscaling 4.1.1 (now including RX 7000/RDNA 3 support), ML Frame Generation 4.0.1, and Ray Regeneration 1.2.0; it preserves analytic fallbacks. **Why it matters:** the target addressable AMD student hardware is materially broader, but AMD now ships separate ML SR and FG networks—the contrast makes OSS's shared-canvas thesis testable. Keep #7's binary-teacher distillation legal/provenance boundary tight.

### Intel / XeSS
- Intel's Arc G-Series handheld announcement positions **XeSS 3** as Super Resolution + Multi-Frame Generation + Xe Low Latency, with multi-frame capability arriving in a handheld-first product line. **Why it matters:** reinforces low-power/handheld latency as a primary student tier and makes Pico/Steam-Deck-class validation more important than desktop-only benchmarks.

### Apple / Metal
- Apple's current Metal page describes a redesigned MetalFX temporal upscaler using Neural Engine plus Neural Accelerators on M5 Pro/Max, along with dynamic-resolution subrectangles, direct motion-vector input, and distortion fields. **Why it matters:** this aligns with OSS's exact-MV/G-buffer contract and provides a concrete Metal port acceptance target for #8; cross-vendor abstraction must not assume CUDA-style dispatch or tensor-core-only inference.

### Game engines
- AMD's FSR 4 UE 5.7 plugin and NVIDIA's DLSS 4.5 UE 5.5–5.8 plugins mean the integration baseline is moving rapidly. OSS needs its own UE capture/runtime surface to be version-resilient rather than an engine-specific fork.

## Action Items / Follow-ups
- [ ] **CRITICAL:** Restore or deliberately terminate `srcnn-v7.0-pico-005`; dashboard evidence says no metric has advanced for 51h, despite "live" copy.
- [ ] Deep-read GS-STVSR's covariance-resampling alignment and adaptive offset window; write an ablation mapping to v7 `V_xt` and disocclusion behavior.
- [ ] Prototype a 2D rigid-region / long-lived-anchor variant inspired by TRiGS, then measure Gaussian-count growth and temporal identity versus current parent-child density.
- [ ] Add velocity-consistency losses from VeloGauss only as an engine-motion-aware ablation; reject if it worsens exact-MV responsiveness or invalidates near-zero-marginal-cost OSS-FX.
- [ ] Complete #12 then #13 before asserting 32 B/Gaussian or finalizing TensorRT FP8 student performance claims.
- [ ] Track DLSS 4.5, FSR SDK 2.3, XeSS 3, and MetalFX M5 as the new competitive/integration baseline; prioritize controlled shared-content DISTS/FLIP/2AFC evaluation over dashboard cross-dataset tables.

## Sources
- [OSS repository and issue tracker](https://github.com/cashcon57/open-supersampling)
- [OSS training dashboard](https://opensupersampling.com)
- [NVIDIA DLSS developer page](https://developer.nvidia.com/rtx/dlss)
- [AMD FSR SDK 2.3](https://gpuopen.com/amd-fsr-sdk/)
- [Intel Arc G-Series / XeSS 3 announcement](https://newsroom.intel.com/client-computing/intel-arc-g-series-processors-set-a-new-standard-for-handheld-pc-gaming)
- [Apple Metal updates](https://developer.apple.com/metal/whats-new/)

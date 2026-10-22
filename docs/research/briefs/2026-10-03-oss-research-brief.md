# OSS Research Brief — 2026-10-03

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only one commit is present in this clone's `git log --oneline -10` output).
- **Open issues:** 12 (#1, #3–10, #12–14). Current technical priorities include #14 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #8 cross-vendor kernels, #4 student distillation, and #6 launch gating.
- **README:** Checked-in snapshot still describes v6/v6.2 and v7 teacher training. It calls v7 Pico-005 the active run, but the live dashboard gives a materially different current status below.
- **Training dashboard:** [opensupersampling.com](https://opensupersampling.com) reports **Trainer offline, stuck at step 36,650 for 342h**; Watcher healthy (published just now), Cloudflare Worker healthy (HTTP 200/86ms), R2 origin healthy (HTTP 200). Active run shown: `srcnn-v7.0-pico-005`, 36.7% of 100K steps, loss 0.19268, estimated 1.1d remaining (estimate is stale/inconsistent with the 342h stall). Best validated row remains v5-pixel-temporal (25.70 dB / 0.1666 LPIPS); host GPU stats unavailable. **Actionable ops concern:** trainer interruption is the top follow-up; dashboard transport is reachable and provides live data, contradicting the previous brief's reported 403.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **[GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting](https://arxiv.org/abs/2604.18047)** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20; arXiv:2604.18047. Flow-guided continuous-time Gaussians, covariance resampling, and adaptive offsets; reports nearly constant inference time across temporal scales and >3× speedup at ×32. **Why it matters:** direct precedent for continuous time/covariance modeling; compare specifically against OSS's native temporal axis and game-MV conditioning. Already in BIBLIOGRAPHY.md.
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** — Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12; arXiv:2605.11489. Combines a differentiable post-rendering super-sampling path with a separate compact temporal interpolation module. **Why it matters:** relevant joint SR/FG quality comparator, but distinguishes itself from OSS by retaining a separate interpolation module. Already in BIBLIOGRAPHY.md.
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** — 2026-08-11; arXiv:2608.10423 (author list not confirmed from retrieved metadata). Proposes non-neural, tiled/layered scene-cache extrapolation with explicit disocclusion handling; paper claims up to 250 Hz at 4K. **Why it matters:** competitive alternative to neural FG and an important control for the claim that a learned canvas is required; compare latency, disocclusion behavior, and required engine buffers. Headline performance is author-reported, not independently validated.

### Super-Resolution / Upscaling
- **[GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution](https://github.com/tljxyys/GaussianSR)** — Jintong Hu et al.; AAAI 2025. Existing arbitrary-scale Gaussian SR baseline surfaced again in repo search; classifier-assigned adaptive kernels and continuous feature field. **Why it matters:** a useful spatial-only baseline for isolating what temporal modeling adds; not a new discovery and already captured in project bibliography.
- **[FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super-Resolution](https://arxiv.org/abs/2510.12747)** — 2025-10-14; arXiv:2510.12747. Streaming one-step distillation, locality-constrained sparse attention, and tiny decoder; reports ~17 FPS at 768×1408 on A100. **Why it matters:** teacher/distillation and locality ideas may transfer, but latency remains far outside OSS's sub-2-ms game budget. Treat as a quality frontier, not a viable student target.

### Frame Extrapolation / Generation
- **Amulet** (above) is the most consequential newly surfaced frame-extrapolation competitor: non-neural sparse layered cache, explicit disocclusions, claimed high-resolution/high-refresh operation. Its scene representation and cache footprint merit a direct architecture comparison.
- No new directly matching single-network SR+extrapolation arXiv work was verified beyond the existing GS-STVSR/3DGS³ comparators.

### Inference Optimization
- No new primary-source, directly applicable TensorRT/INT8/kernel-fusion paper was verified in this sweep. FlashVSR's locality-constrained attention and tiny decoder are potential teacher/student training ideas, not evidence of <2 ms inference.

## Archive.org Findings
- Archive.org site-scoped searches for “GS-STVSR”, “GSASR”, “GRAPE”, “STSS”, and related Gaussian-SR/frame-generation terms yielded irrelevant or generic capture results; no archived paper versions were confirmed. This does not establish absence from the Wayback Machine.
- Queries: [GS-STVSR](https://archive.org/advancedsearch.php?q=GS-STVSR), [GSASR](https://archive.org/advancedsearch.php?q=GSASR), [GRAPE](https://archive.org/advancedsearch.php?q=GRAPE), [STSS](https://archive.org/advancedsearch.php?q=STSS).

## GitHub / Open Source
- **[ContinuousSR](https://github.com/peylnog/ContinuousSR)** (ICLR 2026) surfaced as a recent 2D pixel-to-Gaussian arbitrary-scale SR implementation. Its reported 19.5× speed improvement / +0.90 dB is the authors' benchmark claim; inspect code and hardware conditions before comparison. Opportunity: inspect its Gaussian allocation/covariance strategy as a spatial ablation donor.
- **[OptiScaler](https://github.com/optiscaler/OptiScaler)** remains the relevant cross-vendor middleware/interface project. Its core role is API/backend interchange for existing vendor upscalers, not an OSS-like learned model. No newly verified independent DLSS/FSR analysis tool or directly comparable open-source game upscaler was confirmed.
- GitHub topic searches were noisy and included unrelated or unverified DLSS 5 claims; omitted from evidence-based findings.

## News
- **NVIDIA / DLSS 4.5:** NVIDIA's [GeForce news page](https://www.nvidia.com/en-us/geforce/news/gears-of-war-e-day-dlss-4-5-ray-tracing-rtx-mega-geometry/) (Oct 1) reports *Gears of War: E-Day* launched with Super Resolution, Frame Generation, and Dynamic Multi Frame Generation. Threat: continued bundling and game adoption expand the competitive baseline; NVIDIA's quality/performance claims remain vendor-reported.
- **AMD:** AMD's [FSR technologies page](https://www.amd.com/en/products/graphics/technologies/fidelityfx/super-resolution.html) lists FSR Upscaling 4.1 support for RX 7000/9000 and describes ML-powered upscaling. **Competitive signal:** vendor ML upscaling is extending beyond the newest generation; cross-vendor model portability remains a differentiation opportunity for OSS.
- **Intel:** [XeSS SDK release notes](https://github.com/intel/xess/releases) list XeSS SDK 3.0.2 (July 2026) and 3.0 MFG; Intel's [developer page](https://www.intel.com/content/www/us/en/developer/topic-technology/gamedev/xess.html) says XeSS 3 includes SR, FG/MFG, and low latency. Treat exact hardware availability as SDK-specific; this is a vendor ecosystem threat to OSS integration mindshare.
- **Apple / Metal:** Apple's [Metal What's New](https://developer.apple.com/metal/whats-new/) advertises a redesigned temporal upscaler leveraging Neural Engine and Neural Accelerators on M5 Pro/Max, plus quantized tensor formats. Opportunity/threat: cross-vendor plans need a measured Metal/CoreML lane rather than assuming CUDA quantization maps to Apple hardware.
- **Engine integration:** AMD's [UE FSR plugin guide](https://gpuopen.com/learn/ue-fsr/) (updated Aug 2026) reports support through UE 5.8 and packages ML upscaling, frame generation, and fallback paths in one plugin. This reinforces engine plugin distribution as a competitive moat; OSS's integration path remains unbuilt per README.

## Action Items / Follow-ups
- [ ] Diagnose why `srcnn-v7.0-pico-005` trainer stopped at step 36,650 for 342h; check host/process/logs and resume only if the checkpoint/run is healthy.
- [ ] Reconcile README's stale v6/v6.2 status details with v7 architecture status; dashboard is authoritative for the current run state.
- [ ] Deep-read GS-STVSR and Amulet; benchmark both against OSS on common sequences, reporting frame latency, disocclusion quality, memory, and required inputs.
- [ ] Inspect ContinuousSR and compare Gaussian parameterization/allocation as a spatial-only ablation donor.
- [ ] Track AMD FSR 4.1 expansion, XeSS 3 MFG, DLSS 4.5 adoption, and Apple's temporal MetalFX path; test cross-vendor feasibility against #8.
- [ ] No new bibliography additions this run: identified papers are already represented in BIBLIOGRAPHY.md or are not yet sufficiently verified for bibliographic insertion.

## Sources / Method Notes
- Internal sources: local `git log --oneline -10`, GitHub Issues REST API, README, BIBLIOGRAPHY.md, and live dashboard extraction.
- arXiv records: [GS-STVSR](https://arxiv.org/abs/2604.18047), [3DGS³](https://arxiv.org/abs/2605.11489), [Amulet](https://arxiv.org/abs/2608.10423), [FlashVSR](https://arxiv.org/abs/2510.12747). Author lists/dates stated only where search metadata confirmed them; all paper results are author-reported.
- GitHub/vendor primary pages: [ContinuousSR](https://github.com/peylnog/ContinuousSR), [OptiScaler](https://github.com/optiscaler/OptiScaler), NVIDIA, AMD, Intel, and Apple links in sections above.
- Archive.org searches produced no confirmed paper snapshot; results were noisy. Search coverage is not exhaustive.

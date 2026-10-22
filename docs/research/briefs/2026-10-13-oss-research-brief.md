# OSS Research Brief — 2026-10-13

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke`. The local clone exposes one commit in `git log --oneline -10` (appears shallow/single-commit).
- **Open issues:** 12 total. Notable: #14 VRAM footprint measurement, #13 Nsight L2 profile, #12 packed canvas layout, #10 G-buffer encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher signal, #6 Pico run launch, #5 no-ML shader, #4 student distillation. [Issue tracker](https://github.com/cashcon57/open-supersampling/issues)
- **README snapshot:** top-level narrative describes v7 N-D Gaussian/time-axis teacher training; the model student and cross-vendor kernels remain unbuilt. README still contains older v6.2-oriented status in some subsections.
- **Training dashboard:** active card reports `srcnn-v7.0-pico-005`, step **36,650/100,000**, loss **0.19268**, ETA ~1.1 days; best result remains v5 (25.70 dB PSNR / 0.1666 LPIPS). But trainer is explicitly **offline, stuck at step 36,650 for 582h**, GPU stats unavailable. Watcher, CF Worker (HTTP 200/121ms), and R2 origin report healthy. Treat live card and ETA as stale; training host needs attention.

## New Papers (arXiv)
Search surfaced two material October papers; other broad category searches mostly returned older or tangential works. Metadata checked against arXiv abstract pages.

### Gaussian Splatting / Temporal
- **Sparse-View 4D Gaussian Splatting via Spatiotemporal Priors and Generative Assistance** (Shengqi Wang, Zhengxian Yang, Kaiwen Tian, Yang Liu, Bowen Liu, Hua Du, Taicheng Huang, Jiamin Wu, Tao Yu; submitted 2026-10-03, revised 2026-10-07; [arXiv:2610.04606](https://arxiv.org/abs/2610.04606), SIGGRAPH Asia 2026 Workshops). Region-adaptive Gaussian initialization/densification plus intermediate-time interpolation/flow constraints improves sparse-view dynamic-scene reconstruction; it also adds diffusion pseudo-supervision in unseen views. **Why it matters to OSS:** the mask-voting / region-specific densification and intermediate-time supervision are relevant v7 training ablations, although the diffusion-assisted multi-view reconstruction setup is not a real-time game upscaler.

### Super-Resolution / Upscaling
- **High-Throughput Generative Video Super-Resolution on a Single Consumer GPU** (Georgios Panaretou, Sotirios Chatzis; submitted 2026-10-09; [arXiv:2610.13179](https://arxiv.org/abs/2610.13179)). Zero-shot system combines one-step diffusion SR with flow-based temporal interpolation; reports up to ~32 FPS at 720p on RTX 4090 and up to 44% speedup at 4K from replacing selected high-resolution global-attention layers with neighborhood attention. **Why it matters to OSS:** a perceptual-quality frontier/competitor reference and a concrete attention-localization technique to test in teachers; its heavyweight two-stage approach does not meet the OSS student latency target.

### Frame Extrapolation / Generation
- No new directly applicable arXiv result found in this pass. Amulet (arXiv:2608.10423) remains the recent sparse-layered-cache comparator, documented in yesterday's brief.

### Inference Optimization
- No newly surfaced paper with direct TensorRT/INT8/game-SR deployment evidence. The generative VSR paper above's high-resolution neighborhood attention substitution is the relevant new optimization lead.

## Archive.org Findings
- Broad Archive.org search was noisy and returned old GitHub snapshots/unrelated ML papers. No meaningful new archived copy of GS-STVSR, GSASR, GRAPE, or STSS identified today. Previous Wayback evidence for GS-STVSR is in the 2026-10-12 brief; targeted CDX lookups for GSASR/GRAPE/STSS remain unresolved, not evidence of absence.

## GitHub / Open Source
- Recent-push repository search did not establish a new, validated OSS-like Gaussian-SR implementation. Search results surfaced existing SplatSuRe, GaussianSR, and ContinuousSR projects, but these target 3DGS novel-view reconstruction or single-image arbitrary-scale SR rather than game-frame temporal SR.
- [OptiScaler](https://github.com/optiscaler) remains the relevant open-source interception/technology-swapping project; it is integration-layer competition/adjacency, not a competing trained cross-vendor model. Search did not verify new code/release activity today.
- No new independently verified open-source DLSS/FSR analysis tool surfaced. Treat repository-search results as discovery leads, not audited code or performance claims.

## News
- **NVIDIA / DLSS:** NVIDIA's developer blog documents DLSS 4.5 (Jan 14, 2026): second-generation SR transformer, claimed 5× compute vs prior transformer, plus Dynamic Multi Frame Generation up to 6×; available as plugin/overrides across hundreds of titles. Not a past-7–14-day launch, but this remains the primary quality/compute threat baseline; avoid presenting it as today's breaking news. [NVIDIA source](https://developer.nvidia.com/blog/nvidia-dlss-4-5-delivers-super-resolution-upgrades-and-new-dynamic-multi-frame-generation/)
- **AMD / FSR:** October search result reports AMD plans FSR 4 expansion to APUs, laptops, and handhelds during 2026; secondary-source report, no primary confirmation retrieved. Watch for platform expansion, which narrows OSS's hardware-access differentiator.
- **Intel / XeSS, Apple / Metal, engines:** No well-sourced material development in the recent window surfaced in this search pass. Search results included older/vendor-generic items; no date-verified October announcement to report.

## Action Items / Follow-ups
- [ ] **Operational priority:** investigate why the v7 trainer has been offline/stuck for 570 hours; dashboard freshness makes current run state unreliable.
- [ ] Deep-read arXiv:2610.04606 for region-adaptive density control and intermediate-time supervision; compare its assumptions with exact engine motion vectors.
- [ ] Evaluate the neighborhood-attention substitution from arXiv:2610.13179 as a teacher-side 4K compute ablation; measure quality and end-to-end latency, not just attention-layer speed.
- [ ] Track AMD's reported FSR 4 expansion with primary AMD sources before updating competitive positioning.
- [ ] Revisit Archive.org/Wayback targeted queries for GSASR, GRAPE, and STSS; the current pass did not resolve them.

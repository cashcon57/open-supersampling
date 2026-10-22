# OSS Research Brief — 2026-10-04

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke`; this clone returns only one commit in `git log --oneline -10`.
- **Open issues:** 12. Top visible items include #14 VRAM measurement, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 fusion ablation, #9 compact student reparameterization, and #8 cross-vendor kernels.
- **README:** Status narrative is internally dated/stale: presents v7 Pico-005 as training now, while its broader status table still documents v6/v6.2 work and the v7 text describes a planned 100K-step run. It states the student is not yet trained and DLL-shim/cross-vendor paths are designed, not built.
- **Training dashboard:** `https://opensupersampling.com` returned HTTP 403 to direct fetch in this run; browser rendering was unavailable (local agent-browser permission denied). The Oct 3 brief's last readable state was trainer offline/stuck at step 36,650 for 342h, watcher/origin healthy. Treat that as prior observation, **not verified current telemetry**; dashboard state is unknown today.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **[Affine-Aligned Atlas for Canonical Gaussian Construction in Video Representation](https://arxiv.org/abs/2610.01114)** — Masaya Takabe, Hiroshi Watanabe, Sujun Hong, Tomohiro Ikai; 2026-10-01; arXiv:2610.01114. Proposes affine alignment for canonical Gaussian construction when videos contain substantial global/camera motion. **Why it matters:** potentially useful comparison for separating global camera motion from local primitive motion in a temporal Gaussian representation; video-scene reconstruction rather than real-time game SR, so treat as a low-priority design read.
- **[TSGL: Teacher-Student Graph Learning for 3DGS Compression](https://arxiv.org/abs/2609.38635)** — Matin Bani Saedi, Matthew Kyan, Gene Cheung; 2026-09-29; arXiv:2609.38635. Distills a trained 3DGS representation via graph learning without retraining or original training images. **Why it matters:** possible post-training canvas-compaction donor for #12; verify whether its compression assumptions transfer to 2D/time-indexed, latency-sensitive primitives.

### Super-Resolution / Upscaling
- No newly verified paper in this sweep directly targets game-rendered SR under OSS's real-time constraints. **[ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing](https://arxiv.org/abs/2609.37874)** (Xijun Wang et al.; 2026-09-29) is a fresh adjacent lead: recurrent latent/cache routing reduces dependence on full KV history, but its Wan2.2 diffusion basis is not an evident <2 ms game path. Abstract-level metadata only.

### Frame Extrapolation / Generation
- No fresh arXiv result verified for single-network game SR + extrapolation. Existing material comparators GS-STVSR, 3DGS³, Amulet, and tile-based extrapolation remain the more direct reads; see prior briefs and bibliography.

### Inference Optimization
- No newly surfaced optimization paper met a strong direct-relevance threshold. **NanoVSR** (arXiv:2607.10495; existing bibliography) remains a useful lightweight structural-reparameterization lead, not a new discovery.

## Archive.org Findings
- Queries run through Archive.org Advanced Search for “gaussian splatting” + “super-resolution”, “frame generation” + “deep learning”, and “real-time upscaling” + “game”. Results were noisy/general: first two returned unrelated GitHub/media/OSF records; third returned zero. No archived versions of GS-STVSR, GSASR, GRAPE, or STSS were confirmed. Search coverage is not exhaustive.

## GitHub / Open Source
- Search API queries used `pushed:>=2026-09-20`, sorted by update. **[FoundDSR](https://github.com/yanzq95/FoundDSR)** (updated Sep 25) describes guided 2D Gaussian splatting for *depth* super-resolution, not image/video SR; adjacent representation work only, not a direct competitor.
- **[SE-FG](https://github.com/PhoenixTheSage/SE-FG)** (MIT; updated Oct 1) describes open-source frame generation for Space Engineers. Narrow game-specific integration; inspect only if its motion-vector/render-hook technique is reusable.
- **[dlssg-open](https://github.com/dniminenn/dlssg-open)** (MIT; updated Sep 23) targets NVIDIA DLSS FG on RTX 30 under Proton; integration/compatibility project, not an OSS-like model.
- **[CoreMLVideoUpscaler](https://github.com/koheyeng/CoreMLVideoUpscaler)** (MIT; updated Oct 4) is a macOS Core ML video upscaler with VideoToolbox frame-rate conversion. Potential Metal/CoreML lane reference, but repo metadata alone does not establish quality, speed, or game-render integration.
- No clearly verified new open-source, vendor-neutral game upscaler comparable to OSS. GitHub search is discovery-level; descriptions and update dates were checked, code/performance not validated.

## News
- **NVIDIA / DLSS 4.5 — primary vendor page:** NVIDIA's Oct 1 GeForce update says *Gears of War: E-Day* ships with DLSS 4.5 Super Resolution, Frame Generation, and Dynamic Multi Frame Generation. Threat: another high-profile game bundles the full stack; NVIDIA's performance figures are vendor-reported. [Source](https://www.nvidia.com/en-us/geforce/news/gears-of-war-e-day-dlss-4-5-ray-tracing-rtx-mega-geometry/)
- **AMD FSR — primary developer page:** GPUOpen's current FSR SDK page lists SDK 2.3, FSR Upscaling 4.1.1, Frame Generation 4.0.1, and broader RDNA 3 upscaling availability (page identifies the SDK revision as June 2026; not a new past-14-day release). Competitive signal: AMD continues extending ML upscaling while packaging multiple features in an integration SDK. [Source](https://gpuopen.com/amd-fsr-sdk/)
- **Intel XeSS:** Search surfaced XeSS 3 rollout coverage and SDK 3.0.2 (July) but no clearly verified new primary Intel announcement in the past 14 days. Mark as no fresh verified news, rather than infer a recent release.
- **Apple / Metal, engines, GPU announcements:** no newly verified primary-source item in the past 14 days from this sweep. The CoreMLVideoUpscaler repo is a lead, not a vendor announcement.

## Action Items / Follow-ups
- [ ] Recheck dashboard access/telemetry when reachable; Oct 3's offline/stuck state is last known, not today's measurement.
- [ ] Deep-read TSGL for canvas compaction relevance and benchmark memory/quality trade-offs before adopting.
- [ ] Compare affine-aligned canonicalization against v7 motion/camera handling, but prioritize only if global motion is a demonstrated failure mode.
- [ ] Inspect SE-FG and CoreMLVideoUpscaler for integration/Metal implementation ideas; do not treat repo descriptions as benchmark evidence.
- [ ] Track DLSS 4.5 adoption and AMD FSR SDK 2.3 availability; quantify OSS's integration gap separately from model-quality claims.

## Sources / Method Notes
- Local repo: git log, README, open GitHub Issues API, BIBLIOGRAPHY.md, Oct 3 brief.
- arXiv Atom searches by Gaussian/temporal, frame generation/extrapolation, VSR/distillation, and TensorRT/kernel-fusion terms; metadata/abstract-level summaries, deduplicated by ID. Broad Gaussian queries yielded irrelevant hits, which were excluded. Refresh cross-check on Oct 4: a narrow date-filtered Atom request returned HTTP 429; web search surfaced already-tracked GS-STVSR and the previously listed Oct 1 affine-aligned atlas paper, with no independently verified newer paper. Treat fresh arXiv coverage for this refresh as rate-limited, not evidence of no new submissions.
- Archive.org Advanced Search; GitHub REST repository search; NVIDIA and AMD primary pages; Intel search results treated as secondary/no fresh verification.
- Dashboard direct HTTP returned 403; browser tool failed to launch due to local permission error. No bibliography additions: adjacent papers did not clear the project's direct materiality bar and should be deep-read before citation as architecture inputs.

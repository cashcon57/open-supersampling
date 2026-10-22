# OSS Research Brief — 2026-10-02

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke` (only commit returned by local `git log --oneline -10`). Working tree already contains many untracked dated research briefs and a modified `BIBLIOGRAPHY.md`; preserved them.
- **Open issues:** #14 v6.2-pico-002 VRAM footprint; #13 Nsight L2 hit-rate profiling; #12 packed low-rank canvas layout; #10 decoupled G-buffer encoder ablation; #9 RepVGG/ECBSR/RepSR ablation; #8 cross-vendor kernels; #7 FSR 4 teacher signal; #6 launch on 3080 Ti; #5 no-ML shader; #4 HAT-Tiny distillation.
- **README snapshot:** reports v5 held-out metrics PSNR 25.703 / LPIPS 0.1666 / temporal ratio 0.337× on TartanAir `oldtown`; v6.1 training and v6.2 work described; shipping ≤1M student remains untrained, HAT-Tiny eager timing far over the <2ms target; OSS-FX α<1 path unwired. This is a stale/possibly conflicting architecture snapshot against project context's v7 N-D canvas.
- **Training dashboard:** direct HTTPS fetch returned HTTP 403; browser renderer could not start (local agent-browser binary permission denied). No live run/health facts verified. Follow up by restoring a rendered dashboard path; do not infer status from README.

## New Papers (arXiv)
### Gaussian Splatting / Temporal
- **[Affine-Aligned Atlas for Canonical Gaussian Construction in Video Representation](https://arxiv.org/abs/2610.01114)** (Masaya Takabe, Hiroshi Watanabe, Sujun Hong, Tomohiro Ikai; 2026-10-01) — Uses frame-wise affine transforms to absorb global camera motion before canonical Gaussian construction, reducing deformation burden. **Why it matters:** worth testing affine/global-motion normalization as a cheap stabilizer for the v7 canvas under camera pans; abstract-level only, not game-SR evaluated.
- **[Gaussian Splatting-based Volumetric Video Compression with Sparse 4D Anchors (SAGA)](https://arxiv.org/abs/2609.33969)** (Ge Gao et al.; 2026-09-27) — Sparse anchors encode dynamic Gaussian video and exploit cross-anchor correlations for motion- and texture-dense regions. **Why it matters:** relevant to canvas capacity/memory compression, though video codec rates and quality do not establish real-time game SR suitability; already surfaced in prior research and not appended again.
- **[TSGL: Teacher-Student Graph Learning for 3DGS Compression](https://arxiv.org/abs/2609.38635)** (Matin Bani Saedi, Matthew Kyan, Gene Cheung; 2026-09-29) — Learns a local geometry graph and transforms Gaussian attributes into compact low-frequency coefficients. **Why it matters:** potential packed-canvas compression inspiration for #12/#14; not directly validated on time-varying game canvases.

### Super-Resolution / Upscaling
- **[ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing](https://arxiv.org/abs/2609.37831)** (Xijun Wang, Xin Li, Suhang Yao, Zirui Lang, Bingchen Li, Zhibo Chen; 2026-09-29) — Streaming diffusion VSR uses recycled latents and layer-specific temporal cache scopes. **Why it matters:** cache-budget/scheduling ideas may aid teacher training or evaluation, but its Wan2.2 diffusion basis is not a plausible <2ms student.
- **[NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices](https://arxiv.org/abs/2607.10495)** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11) — Fully convolutional VSR uses structural reparameterization for standard-convolution deployment and learns implicit temporal alignment. **Why it matters:** directly matches the student deployment constraint and #9 reparameterization ablation; investigate benchmarks and license. Abstract-level summary.

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (Sebastian Künzel et al.; 2026-08-11) — Sparse layered image-space cache stores potentially visible geometry and adaptively refreshes stale/missing shading. **Why it matters:** strong alternative/competitor to simple canvas warping for disocclusion handling; compare quality, storage, and refresh cost against OSS-FX. Already tracked in bibliography.

### Inference Optimization
- No newly identified primary-metadata paper in this sweep that directly addresses game-SR TensorRT/INT8 or fused kernels beyond NanoVSR's inference reparameterization above.

## Archive.org Findings
- Advanced Search for `"gaussian splatting" super-resolution`: 3 records, sampled results were GitHub snapshots, not archived papers. No confirmed archived versions of GS-STVSR, GSASR, GRAPE, or STSS in this probe. Search scope is limited; no inference of removal.

## GitHub / Open Source
- GitHub REST repo search (updated/pushed since 2026-09-18) surfaced **[Nicodemo-Campos/OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen)**, a new cross-platform Vulkan frame-generation/interpolation/upscaling framework (MPL-2.0; 0 stars; pushed Sep 30). **Threat/watch:** broad feature overlap with OSS, but early-stage and no evidence yet of quality/performance. Audit source and runnable benchmarks before positioning claims.
- **[Adam-AtlasSoftware/RDNU](https://github.com/Adam-AtlasSoftware/RDNU)** (GPL-3.0; 15 stars; pushed Sep 24) describes an RDNA3-focused ML upscaler trained on game-image datasets. **Opportunity/competitor:** hardware-specific student/integration comparator; inspect license implications, code, and reproducibility before borrowing or benchmarking.
- Search also found FoundDSR (depth SR) and ResonantBraneSplatting (arbitrary-scale SR), both tiny and not clearly game/temporal focused. GitHub search is discovery, not source audit.
- No new developer-blog technical release independently confirmed in this sweep.

## News
- Broad web search returned mostly mixed-quality and unrelated headlines; no last-14-day DLSS/FSR/XeSS/Apple/engine claim was independently confirmed against a primary vendor changelog/page during this run. Prior dated briefs contain vendor announcements but are not evidence of a new development today. No unsupported news claims added.

## Action Items / Follow-ups
- [ ] Restore rendered dashboard access and capture exact active run, step/loss, and trainer/watcher health; current HTTPS returns 403.
- [ ] Audit OpenFrameGen maturity, license, tested hardware, quality metrics, and input/API assumptions; compare to OSS integration scope.
- [ ] Compare NanoVSR's reparameterized conv student and implicit temporal alignment against the ≤1M student and #9 ablation.
- [ ] Prototype global affine-motion compensation / normalization for v7 camera-motion sequences; measure whether it reduces canvas deformation burden.
- [ ] Evaluate TSGL/SAGA compression ideas against packed layout and measured VRAM/L2 (#12/#13/#14), not reported codec metrics.
- [ ] Reconcile README's v6/v6.2 snapshot with the v7 project context.

## Sources / Method Notes
- Local git log/status, README, GitHub Issues REST API; dashboard fetch attempted at `https://opensupersampling.com` and browser render attempted, both blocked as noted.
- arXiv Atom API queries: Gaussian-video, VSR/real-time/distillation, frame extrapolation/interpolation, TensorRT/kernel fusion + SR. Summaries based on arXiv metadata/abstracts; unrelated query results filtered.
- Archive.org Advanced Search query `"gaussian splatting" super-resolution`; sampled 3 records.
- GitHub REST Search API queries with `pushed:>=2026-09-18`, sorted by updated; metadata reported for candidates, no source audit.
- Web search used for news leads; no recent primary-source claim verified.

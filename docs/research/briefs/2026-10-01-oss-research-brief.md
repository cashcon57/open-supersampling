# OSS Research Brief — 2026-10-01

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke`; local `git log --oneline -10` contains only this commit.
- **Open issues:** 12. Visible priorities: #14 VRAM footprint, #13 Nsight L2 profile, #12 packed low-rank canvas, #10 encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher signal, #6 launch, #4 HAT-Tiny distillation.
- **README:** Pre-alpha snapshot still describes v6/v6.2 as current; student not yet trained, contradicting current v7 research framing.
- **Training dashboard:** Rendered `https://opensupersampling.com` says active `srcnn-v7.0-pico-005`, step 36,650/100,000, loss 0.19268, 36.7%, ETA ~1.1d. Contradiction: headline says “live”, but health says **Trainer — offline**, stuck at step 36650 for **294h**; dashboard GPU stats say unavailable/host offline or rsync failed. Watcher healthy (published 1s ago), CF Worker healthy (HTTP 200/157ms), R2 origin healthy (HTTP 200). The live vs offline contradiction, unchanged step/loss, and implausibly accumulated stuck duration need host-side verification; healthy publishing is not evidence of training.

## New Papers (arXiv)
Abstract/metadata-level review; most candidates are older/newly surfaced rather than newly published today.

### Gaussian Splatting / Temporal
- **[SAGA: Gaussian Splatting-based Volumetric Video Compression with Sparse 4D Anchors](https://arxiv.org/html/2609.33969v1)** (arXiv:2609.33969; 2026-09-27; author metadata not independently retrieved) — Sparse hierarchical 4D anchors decode finer Gaussian primitives via offsets and compact anchor-local hash residuals. **Why it matters:** promising packed/hierarchical temporal-canvas storage and bandwidth design analogy for OSS #12, but compression results do not establish game SR quality or latency.
- **[GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting](https://arxiv.org/abs/2604.18047)** (Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha; 2026-04-20) — Continuous Gaussian motion with covariance-resampling alignment and adaptive motion offsets; reports near-constant inference at ×2–×8 temporal scales. **Why it matters:** direct design comparator for v7 covariance/time evolution; already tracked in bibliography.

### Super-Resolution / Upscaling
- **[RelayVSR: Large-Small Model Collaboration for Efficient Real-World Video Super-Resolution](https://arxiv.org/abs/2609.37850)** (Xijun Wang, Xin Li, Zirui Lang, Suhang Yao, Haoran Li, Zhibo Chen; 2026-09-29) — Large/small model collaboration for real-world VSR. **Why it matters:** possible teacher-to-student training and selective refinement pattern; evaluate only if its large model can be kept off the runtime path.
- **[ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing](https://arxiv.org/abs/2609.37831)** (Xijun Wang, Xin Li, Suhang Yao, Zirui Lang, Bingchen Li, Zhibo Chen; 2026-09-29) — Streaming diffusion VSR learns layer-specific temporal KV-cache scopes and recycles predicted latents. **Why it matters:** cache-budget/scheduling ideas may inform offline teacher training, but Wan2.2 diffusion is not a plausible <2 ms student. Both papers are too new/abstract-level for bibliography addition this run.

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (Sebastian Künzel et al.; 2026-08-11) — Sparse tiled layered cache supports non-neural extrapolation and localized refresh, reporting up to 250 Hz at 4K. **Why it matters:** direct competitor/alternative to OSS-FX's canvas warp; layered disocclusion handling is a valuable comparison. Previously tracked.
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** (Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu; 2026-05-12) — Combines gradient-guided SR with a compact recurrent interpolation module. **Why it matters:** external unified SR+temporal rendering comparator, though it uses separate branches rather than OSS's single canvas; already tracked.

### Inference Optimization
- No new primary-metadata-verified paper found this sweep specifically on game-SR TensorRT/INT8 or kernel fusion. Continue prioritizing measured VRAM/L2 and end-to-end timing (#12/#13) over generic optimization claims.

## Archive.org Findings
- Archive.org Advanced Search phrase counts/results: Gaussian SR 3 (sampled items are GitHub snapshots), frame generation/deep learning 244 (sampled records irrelevant), real-time game upscaling 107 (sampled records unrelated). No confirmed archived versions of GS-STVSR, GSASR, GRAPE, or STSS; broad keyword counts are noisy, not evidence of removal.

## GitHub / Open Source
- **Opportunity/competitive watch — [Open-Frame-gen](https://github.com/Deadly-pro/Open-Frame-gen):** search surfaced a claimed ~90K-parameter CNN plus TensorRT FP16/C++ capture-overlay path, 100–200 FPS at 1080p on RTX 3050 Laptop, <200MB VRAM, with released weights and training/benchmark files. Promising student/integration comparator, but self-reported claims; audit reproducibility, quality metrics and licensing before comparison. Distinct from similarly named OpenFrameGen search lead; no GitHub API metadata verified for this repo.
- **Open-source middleware watch — [OptiScaler](https://github.com/optiscaler/OptiScaler):** surfaced as cross-GPU upscaler/frame-gen replacement/modding layer (GPL-3.0 per search result); adjacent integration ecosystem, not a competing learned model. Treat as search lead; no latest commit audit this run.
- NVIDIA primary Gamescom 2026 page confirms DLSS 4.5 Ray Reconstruction's second-generation transformer and broad RTX availability; AMD's official technology page confirms FSR Upscaling 4.1 is available on RX 7000/9000, with RX 6000 support planned in 2027. These hardware/vendor-locked advances strengthen OSS's cross-vendor differentiation, while raising the quality bar.
- Intel's official XeSS page describes XeSS 3 combining SR, Multi-Frame Generation and Low Latency; Intel September title support indicates deployment momentum. No specific newly dated XeSS technical release independently verified.

## News
- **NVIDIA (primary-confirmed):** Gamescom 2026 announcement makes DLSS 4.5 Ray Reconstruction available with a second-generation transformer; Sept 29 CONTROL Resonant announcement describes its integration with path tracing and DLSS 4.5. Competitive threat in the fidelity/temporal-reconstruction stack, though RTX-specific and not OSS's open engine-neutral architecture.
- **AMD (primary-confirmed):** FSR Upscaling 4.1 now supports RX 7000/9000; RX 6000 support planned for 2027. AMD frames Redstone as ML-based upscaling, frame generation, ray regeneration and radiance caching. Opportunity for OSS cross-vendor positioning; threat to quality expectations.
- **Intel (official product page):** XeSS 3 advertises SR + multi-frame generation + low latency; no confirmed new September algorithm details.
- **Apple/Metal and engine updates:** No primary-source recent change verified. Search results were noisy; no claim of absence beyond this sweep.

## Action Items / Follow-ups
- [ ] Verify trainer host process/metrics against dashboard's “live” vs offline state; step 36,650/loss 0.19268 unchanged and displayed stuck duration is 294h.
- [ ] Compare SAGA hierarchical sparse 4D anchors/hash residuals and GS-STVSR covariance alignment with v7 memory/time axes.
- [ ] Audit Open-Frame-gen weights, benchmark protocol, license and latency/quality on comparable inputs; use as student-tier comparator if reproducible.
- [ ] Track RelayVSR/ReCaVSR for distillation/cache-budget ideas, not direct student inference.
- [ ] Reconcile README's v6/v6.2 snapshot with v7 status; inspect and preserve existing bibliography edits (no additions made in this run).

## Sources / Method Notes
- Local git log, GitHub open-issues API, README and bibliography; dashboard rendered from `https://opensupersampling.com`.
- arXiv discovery via web search and arXiv pages/metadata for identified papers; broad Atom queries were not available in this run. Abstract-level summaries; author metadata for SAGA not retrieved.
- Archive.org Advanced Search phrase probes (sampled results, not exhaustive paper archive test).
- GitHub/source monitoring via web search; repo metadata/code not audited for Open-Frame-gen or OptiScaler.
- Vendor claims checked against NVIDIA GeForce Gamescom 2026 pages, AMD FSR technologies page and Intel XeSS Gaming page. Non-primary news leads excluded.

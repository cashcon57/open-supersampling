# OSS Research Brief — 2026-09-28

## Project Status Summary
- Latest commits: `b44278a test: add local Chrome Playwright dashboard smoke` (only commit in local `git log --oneline -10`; shallow/limited local history).
- Open issues (12): #14 VRAM footprint, #13 Nsight L2 hit-rate, #12 packed low-rank canvas, #10/#9 v6.2.1 ablations, #8 cross-vendor kernels, #7 FSR 4 teacher signal, #6 Pico launch, #5 no-ML shader, #4 HAT student. API returned 12; first 10 listed.
- README is internally stale: currently describes v7 N-D time-axis training in the status snapshot, but many later sections still describe v6/v6.2 plans. It says ≤1M-param student is not trained and measured v5 quality remains the best recorded.
- Dashboard (rendered/extracted): `srcnn-v7.0-pico-005` labeled live at step 36,650/100K, loss 0.19268, ETA ~1.1d; Trainer tooltip says **offline, stuck at step 36,650 for 222h**. Watcher healthy (published 1s ago), CF Worker HTTP 200/130ms, R2 HTTP 200. Same step/loss as prior report; increased stuck duration, no evidence of training progress. Live-vs-offline remains an operational contradiction.
- Refreshed README indicates current architecture v7 N-D Gaussian canvas with native time axis, while commit history is shallow and has only `b44278a` locally. README still contains conflicting older v6/v6.2 prose.

## New Papers (arXiv)
Search was targeted web/arXiv-page lookup rather than complete Atom/API category sweeps; summaries are abstract-level. “New” means newly surfaced as material, not newly published.

### Gaussian Splatting / Temporal
- **GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting** — Mingyu Shi, Xin Di, Long Peng, Boxiang Cao, Anran Wu, Zhanfeng Feng, Jiaming Guo, Renjing Pei, Xueyang Fu, Yang Cao, Zhengjun Zha (2026-04-20; [arXiv:2604.18047](https://arxiv.org/abs/2604.18047); already in bibliography). Abstract reports covariance-resampling alignment and nearly constant inference over temporal factors ×2–×8; validate against OSS's native-time covariance assumptions.
- **3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** — Yibo Zhao, Fan Gao, Youcheng Cai, Ligang Liu (ECCV 2026 poster, presented Sep 10; [poster/abstract](https://eccv.ecva.net/virtual/2026/poster/3945), paper linked there; arXiv ID listed in existing bibliography as 2605.11489). Combines gradient-aware SR and a separate lightweight temporal interpolation network over low-resolution 3DGS output. Competitive threat/benchmark, but repository currently empty; it does not substantiate performance/code claims yet. The split modules contrast with OSS's shared persistent canvas.
- **Adaptive Densification for High-Fidelity and Efficient Sparse Gaussian Splatting in Arbitrary-Scale Super-Resolution (QuADA-GS)** — Giulio Federico, Giuseppe Amato, Claudio Gennaro, Fabio Carrara, Marco Di Benedetto ([arXiv:2606.29400](https://arxiv.org/abs/2606.29400), v2 revised Sep 8). Abstract proposes LR-conditioned content-adaptive sparse Gaussian allocation and sparse communication to balance fidelity/cost. Direct opportunity for v7 bounded-canvas adaptive spawning and #12 memory/latency ablations; bibliography already has this arXiv ID under an alternate title, avoid duplicate append until title/version is reconciled.

### Super-Resolution / Upscaling
- **Compressed-Domain-Aware Online Video Super-Resolution** — Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, Xiaoyao Yang (2026-03-08; [arXiv:2603.07694](https://arxiv.org/abs/2603.07694), CVPR 2026). Uses codec motion vectors/residual maps/frame types, MV coarse warp plus learned local offsets and residual-gated fusion; abstract reports >2× speed and up to +0.13 dB on REDS4 vs TMP. **Opportunity:** ablate codec/MV reliability and residual confidence as inputs/gating alongside exact engine vectors; streaming-video gains are not game-latency proof.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg (2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423)). Explicitly caches potentially visible layered geometry/shading to handle disocclusions without hallucination and schedules tile refresh; claims up to 250 Hz at 4K. **Threat/opportunity:** close conceptual overlap with OSS's persistent image-space canvas and extrapolation; deep-read whether its non-neural explicit cache and disocclusion handling outperform a Gaussian representation, and compare actual hardware/protocol before accepting headline rate.

### Inference Optimization
- No newly verified task-specific TensorRT/INT8/kernel-fusion paper located in this targeted pass.

## Archive.org Findings
- Archive.org: broad search surfaced no reliable confirmed paper snapshots; Google-indexed lookups mostly rediscovered live arXiv pages and the local repo's prior survey. Current direct archive.org/API attempt was blocked by the Python TLS certificate chain in this runtime, so the requested three keyword searches and targeted Wayback CDX checks remain unverified. No claim that archived copies are absent.


## GitHub / Open Source
- **Threat/adjacent project:** [OpenFrameGen](https://github.com/Nicodemo-Campos/OpenFrameGen) advertises cross-platform open frame generation/interpolation/upscaling on Linux/Windows with Vulkan and modular backends; GitHub search metadata: MPL-2.0, pushed 2026-09-27. Audit actual implementation, supported inputs and measured quality/latency; repository description alone does not establish parity.
- **Adjacent implementation:** [RDNU](https://github.com/Adam-AtlasSoftware/RDNU) is described as an RDNA3 ML upscaler using decoupled G-buffer guidance; pushed Sep 24, GPL-3.0. Comparative lead only; license incompatible with direct Apache-2.0 code reuse absent permission.
- **Tooling/modding:** [dlss-unlocked](https://github.com/ShyVortex/dlss-unlocked) and [DLSS5-FrameGen](https://github.com/ztwzeroo/DLSS5-FrameGen) surfaced as experimental feature-unlock/modding projects. Not independent OSS upscaler competitors; audit claims and safety/legal boundaries before referencing technically.
- Search surfaced neural upscaler add-on [ls-addon-manager](https://github.com/Echo-Storm/ls-addon-manager); repo metadata says MIT and updated Sep 28. This is an integration/tooling signal, not a model.
- GitHub Search API returned two Gaussian-SR hits, but malformed/missing license metadata interrupted per-repo inspection; no new Gaussian-SR conclusion. Search sample is not exhaustive.

## News
- **NVIDIA primary source:** [Sep 22 developer update](https://developer.nvidia.com/blog/whats-new-for-game-developers-dlss-5-with-3d-guided-neural-rendering-nvidia-ace-updates-and-new-rtx-kit-capabilities/) promotes DLSS 5 3D-guided neural rendering, not simply a conventional SR/frame-generation update. Competitive positioning: increasingly game-guided neural rendering and tooling.
- **AMD primary source:** [FSR technologies](https://www.amd.com/en/products/graphics/technologies/fidelityfx/super-resolution.html) says FSR Upscaling 4.1 is available on RX 7000 as well as 9000; Redstone includes ML upscaling, frame generation, ray regeneration, and radiance caching. This meaningfully broadens ML upscaling beyond latest AMD generation, raising cross-vendor pressure.
- **Apple primary source:** [Metal “What’s New”](https://developer.apple.com/metal/whats-new/) describes redesigned temporal MetalFX upscaler using Neural Engine and Neural Accelerators on M5 Pro/Max, lower render resolutions, subrectangles, motion-vector input. Opportunity and threat for Metal-specific targets; reinforces need to scope cross-vendor performance honestly.
- Intel search surfaced XeSS SDK 3.0.2 dated Jul 24 and a public-header license change Sep 9; no new Sep 14–28 feature release verified.
- No additional primary-source-confirmed vendor/engine release established in this refresh. Web results included secondary claims about DLSS 5/FSR/XeSS; excluded from this dated update pending vendor confirmation. Previous verified NVIDIA/AMD/Apple items above remain leads, not new changes in this run.


## Action Items / Follow-ups
- [ ] Investigate dashboard contradiction: verify trainer process/metrics at source; 222h-stuck tooltip versus live badge and fresh watcher publication.

- [ ] Deep-read Amulet; compare layered-cache disocclusion solution, memory/cost and extrapolation quality against v7's native-time Gaussian canvas.
- [ ] Prototype QuADA-GS-style LR-conditioned adaptive Gaussian allocation; measure VRAM, sparse communication overhead, and quality against fixed-density canvas.
- [ ] Track 3DGS³ paper/repository release and obtain its paper PDF/benchmarks; repository is currently empty.

- [ ] Audit OpenFrameGen implementation and benchmarks; assess whether modular Vulkan integration competes with OSS scope/positioning.
- [ ] Test CDA-VSR-inspired motion-vector confidence/residual gating in an ablation; codec vectors are only a proxy for engine G-buffers.
- [ ] Update README's conflicting architecture/status sections and distinguish actual measured student/runtime from targets.
- [ ] Run targeted Wayback/CDX checks for GS-STVSR, GSASR, GRAPE, and STSS.

Sources: local repo README/git/issues; rendered/extracted dashboard; arXiv abstract pages; GitHub Search API/repository pages; Archive.org Advanced Search; NVIDIA, AMD, Apple primary developer pages. Search coverage is targeted, not exhaustive. Dashboard text reflects the rendered/extracted page; browser executable itself failed to launch due permission denied.

# OSS Research Brief — 2026-10-12

## Project Status Summary
- **Latest commits:** `b44278a test: add local Chrome Playwright dashboard smoke`; `git log --oneline -10` yielded only this commit (shallow/single-commit checkout).
- **Open issues:** 12 total; first ten: #14 v6.2 VRAM measurement, #13 Nsight L2 profile, #12 packed canvas layout, #10 G-buffer encoder ablation, #9 student reparameterization, #8 cross-vendor ports, #7 FSR 4 teacher signal, #6 pico launch, #5 no-ML shader, #4 HAT-Tiny distillation. [Open issues](https://github.com/cashcon57/open-supersampling/issues)
- **README:** describes v7.0 N-D Gaussian/time-axis teacher run `srcnn-v7.0-pico-005` as training; shipping student not yet trained. README's static tier panel is stale (v6.2) relative to the current top-level project narrative.
- **Training dashboard (rendered 2026-10-12):** shows v7.0 Pico OSS-FX at step **36,650/100,000**, loss **0.19268**, viz step 36,000, ETA ~1.1d; best held-out baseline remains v5 (PSNR 25.70 / LPIPS 0.1666). Health tooltip simultaneously says trainer **offline, stuck at step 36,650 for 558h**, GPU stats unavailable; watcher (published 0s ago), CF Worker (HTTP 200/113ms), and R2 origin (HTTP 200) healthy. Dashboard advertises “live” despite offline trainer and unchanged metrics; the static teacher/student panel still says v6.2-pico-002. Treat live card/ETA as stale published data, not evidence of active training. Direct HTTP to `opensupersampling.com` returned 403; browser-rendered canonical redirect to `opensupersampling.org` exposed the state.

## New Papers (arXiv)
Search used arXiv Atom (Oct 12, 2026); abstracts/metadata only. Targeted broad queries returned many unrelated results, filtered for OSS relevance; not exhaustive.

### Gaussian Splatting / Temporal
- **SteadySplats: Resampling of Low-Variance Gaussians for High-Fidelity Stochastic Rendering** (Felix Windisch, Thomas Köhler, Lukas Radl, Chris Wyman, Georgios Kopanas, Bernhard Kerbl, Markus Steinberger; 2026-10-04; [arXiv:2610.05576](https://arxiv.org/abs/2610.05576)). History-based spatial resampling and temporal importance resampling reduce stochastic-render noise coherently under camera motion; **opportunity:** inspect temporal resampling/variance control as a possible stability technique for the Gaussian canvas, while noting its target is stochastic 3DGS rendering rather than game SR.
- **Mobile-4DGS: Unified Static-Dynamic Real-time Mobile Gaussian Splatting** (Xiaobiao Du, Beixi Hao, Zhen Fang, Tianqing Zhu, Richard Hartley, Xin Yu; 2026-10-04; [arXiv:2610.05289](https://arxiv.org/abs/2610.05289)). Compact real-time static/dynamic Gaussian rendering with SH compression and pre-baked appearance offsets; **opportunity:** review representation/attribute compression ideas against packed canvas bandwidth and mobile tier constraints; not SR evidence.
- **Efficient Gaussian Splatting Sequence Compression with Standard Video Codecs** (Qi Yang, Shuting Xia, Le Yang, Geert Van Der Auwera, Zhu Li; 2026-10-06; [arXiv:2610.07795](https://arxiv.org/abs/2610.07795)). Inter-PLAS improves inter-frame correlation before conventional video-codec compression of Gaussian sequences; **opportunity:** possible packed-canvas/checkpoint storage approach only if profiling identifies bandwidth/storage pressure, not a runtime renderer replacement.

### Super-Resolution / Upscaling
- **High-Throughput Generative Video Super-Resolution on a Single Consumer GPU** (Georgios Panaretou, Sotirios Chatzis; 2026-10-09; [arXiv:2610.13179](https://arxiv.org/abs/2610.13179)). Zero-shot pipeline pairs time-distilled one-step diffusion upscaling with a flow interpolator for temporal stabilization; **positioning signal:** quality-frontier competitor, but separate heavyweight spatial and temporal stages do not match OSS's <2 ms unified-canvas goal.

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; [arXiv:2608.10423](https://arxiv.org/abs/2608.10423)). Predictively fills a sparse tiled/layered image-space cache with potentially visible geometry/shading, then synthesizes future views while refreshing missing/stale tiles; **direct threat/comparator:** GPU rendering-side extrapolation tackles disocclusion by caching geometry rather than hallucinating. Not a direct game-MV Gaussian SR replacement, but deep-read its latency, required scene access, and tile lifetime scheduler against OSS-FX.

### Inference Optimization
- **NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices** (Filip Pawlicki, Marcel Kańduła, Marcin Pucek, Kamil Dobies; 2026-07-11; [arXiv:2607.10495](https://arxiv.org/abs/2607.10495)). Fully convolutional temporal model uses structural reparameterization to fuse training-time branches into standard inference convolutions; **opportunity:** directly test fused-graph student distillation/export for TensorRT and cross-vendor kernel simplicity.
- **Streaming-Aware Diffusion for Real-Time Video Super-Resolution via Cross-Step Attention** (Harris Partaourides, Sotirios Chatzis; 2026-10-08; [arXiv:2610.11746](https://arxiv.org/abs/2610.11746)). Reuses denoising features/states across adjacent frames/steps; **opportunity:** feature-cache/reuse ideas may inform teacher supervision or state design, but iterative diffusion is a poor direct fit for game latency.

## Archive.org Findings
- Broad `advancedsearch.php` results were noisy: “gaussian splatting super-resolution” returned 3 unrelated GitHub/listing snapshots; “frame generation deep learning” returned 249 mixed records; “real-time upscaling game” returned 109 mixed records. These counts are not evidence that papers are archived or absent.
- Targeted Wayback CDX confirms archived HTTP-200 snapshots for [GS-STVSR arXiv abstract](https://web.archive.org/web/20260421052653/https://arxiv.org/abs/2604.18047) (several snapshots on 2026-04-21). CDX query for GSASR/GRAPE/STSS patterns timed out; no conclusion on those titles.

## GitHub / Open Source
- **Open-source FG opportunity / competitive signal — [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG):** REST search metadata: Apache-2.0, 8 stars, last pushed Oct 9; described as experimental Vulkan/mobile platform-agnostic frame-generation V1 POC. Inspect its actual inputs/quality/latency before treating it as usable or directly comparable.
- **Potential integration/positioning threat — [DLSS Unlocked](https://github.com/ShyVortex/dlss-unlocked):** search metadata shows 808 stars and recent activity, describing DLSS SR/MFG enablement on RTX 20/30; it exposes vendor models more broadly rather than offering an OSS-trained cross-vendor model.
- **HelixSR-Manager** (Oct 12 update; metadata: 2 stars, license unverified) presents a manager/compiler GUI for HelixSR, a DLSS Model E compute-shader upscaler on AMD RDNA; worth checking as a community portability signal, but tiny/early project, not verified as a direct OSS competitor.
- No fresh Gaussian-SR repository surfaced in the recent-push GitHub search. Queries used unquoted repository search plus `pushed:>=2026-09-28`; search metadata is not a code/benchmark audit.

## News
- **FSR 4 / handhelds (RSS lead, unconfirmed):** Google News RSS dated Oct 7–10 reports AMD confirmation of FSR 4 on APUs/handhelds by end of 2026 and a lightweight model. Search results point to third-party outlets, not an opened AMD primary announcement; treat as a lead, not verified roadmap. If confirmed, raises relevance of OSS Pico/mobile tier.
- **DLSS 4.5 on Radeon (unverified/experimental):** RSS surfaced Wccftech’s Oct 1 claim of a mod using RDNA4 FP8; not primary-source-confirmed and outside 7–14-day window (included as adjacent signal only). Do not treat as vendor support.
- **Intel/XeSS and Apple/Metal:** no high-confidence, directly relevant new primary-source update found in this sweep; news RSS results were largely tangential.
- **Engines:** no directly relevant Unreal/Unity upscaling news verified in the queried RSS results.

### Run refresh — 2026-10-12
- Dashboard re-render confirms the stale-state contradiction: headline/card call v7.0 live at step 36,650, loss 0.19268, viz 36,000 and GPU stats unavailable, while the Trainer tooltip says offline and stuck at the same step for **558h**. Watcher published 1s ago; CF Worker HTTP 200/97ms; R2 HTTP 200. The data publisher is functioning; training is not demonstrably progressing. [Dashboard](https://opensupersampling.org/)
- **Open-source implementation lead:** [ContinuousSR reproduction](https://github.com/peylnog/ContinuousSR/) says the original pretrained model/code cannot be released, but provides a community reproduction and pretrained weights, with non-commercial CC BY-NC 4.0 terms. The underlying Pixel-to-Gaussian paper is already in `BIBLIOGRAPHY.md`; assess as a reproducibility/quality comparator only, not a permissively licensed dependency.
- **Vendor signal:** NVIDIA's developer blog search surfaces DLSS 5 / 3D-Guided Neural Rendering and an Oct 12 post (ReplaySSM is unrelated). This is a material competitive direction but is not an OSS-style vendor-neutral SR architecture; the visible DLSS 5 developer post was dated Sep 22, outside the 7–14 day window. [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- Focused Archive.org advanced search for GS-STVSR/GSASR/GRAPE/STSS returned **0 records**; this is a metadata search result, not proof no archived snapshots exist. Earlier CDX check in this memo confirmed GS-STVSR snapshots; the remaining titles still need targeted CDX checks.
- No bibliography append: the newly considered ContinuousSR reference is already listed; other papers from this report are already present in the same-day bibliography section.

## Action Items / Follow-ups
- [ ] **Ops first:** confirm trainer process/metrics heartbeat with host owner; dashboard is frozen at step 36,650 with 558h offline tooltip despite “live” card and healthy publishing services.
- [ ] Deep-read Amulet (arXiv:2608.10423): compare future-visible cache, disocclusion handling, and adaptive tile scheduler with OSS-FX; record access/latency assumptions.
- [ ] Evaluate NanoVSR structural reparameterization as a student export baseline; benchmark fused graph end-to-end across runtimes.
- [ ] Inspect SteadySplats temporal resampling for a potential stability/variance-control ablation; avoid assuming stochastic 3DGS results transfer to game inputs.
- [ ] Check ZeroFG and HelixSR-Manager code maturity and reproducible supported-input/latency evidence.
- [ ] Re-run focused Wayback CDX checks for GSASR, GRAPE, and STSS; current CDX query timed out.
- [ ] Verify FSR 4 handheld/APU roadmap against AMD primary sources before changing competitive assumptions.

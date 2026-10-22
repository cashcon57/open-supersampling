# OSS Research Brief — 2026-08-17

## Project Status Summary
- **Latest commits:** this shallow checkout exposes only `b44278a test: add local Chrome Playwright dashboard smoke`; working tree already contains the 2026-08-11–16 briefs and a pre-existing `BIBLIOGRAPHY.md` update.
- **Open issues:** 12. The critical dependency sequence is #12 packed low-rank canvas → #13 Nsight L2 profile → #14 VRAM measurement; #4/#9 student distillation/reparameterization, #7 FSR 4 teacher signal, #8 non-CUDA kernels, and #10 G-buffer ablation remain open.
- **README snapshot:** v7 is the active N-D `(x,y,t)` Gaussian teacher with `V_xt` cross-correlation; the ≤1M shipping student and HIP/Metal/Vulkan/Level Zero paths are still unbuilt.
- **Training dashboard:** Trainer is **offline at step 36,650 for 194h**; watcher, Worker, and R2 are healthy.[1] This is 24h beyond yesterday with no step advance. Dashboard UI also shows a stale `v6.1 Pico`/0-step tile while README names `srcnn-v7.0-pico-005`; treat dashboard run labeling as inconsistent until metrics publishing is repaired.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **[ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields](https://arxiv.org/abs/2608.04581)** (arXiv:2608.04581; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko; 2026-08-05; already tracked) — dynamically assigns Gaussian count and feature-channel budget per anchor from local spatiotemporal complexity. **Why it matters:** directly motivates a bounded parent/child canvas experiment for #12, with temporal-identity and L2-residency gates.[2]
- **[Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging](https://arxiv.org/abs/2608.10712)** (arXiv:2608.10712; Tim-Felix Fassch, Jochen Kall, Cyrill Stachniss; 2026-08-11; already tracked) — saliency-guided merging and controllable LoD reduce redundant feed-forward Gaussian sets. **Why it matters:** candidate post-spawn compaction mechanism; test without losing temporal primitive identity.[4]

### Super-Resolution / Upscaling
- **[MeanSR: Restoration Trajectory Learning for One-Step Perceptual Super-Resolution](https://arxiv.org/abs/2608.09405)** (arXiv:2608.09405; Axi Niu, Jiawei Kou, Kang Zhang, Qingsen Yan, Jinqiu Sun, Yanning Zhang; 2026-08-10; newly material) — learns an LR-conditioned average restoration-velocity field for one-step perceptual SR rather than relying on an expensive pretrained-teacher distillation route. **Why it matters:** a teacher-free control for the HAT→≤1M student plan, but not evidence of game-temporal or <2ms performance.[3]
- No fresh primary TensorRT/kernel-fusion paper cleared the materiality threshold. The existing fused-deploy-graph QAT references remain downstream of #12→#13→#14 measurement data.

### Frame Extrapolation / Generation
- **[Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading](https://arxiv.org/abs/2608.10423)** (arXiv:2608.10423; Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; already tracked) — uses a sparse layered image-space cache, predictive shading, and adaptive stale-tile refresh for multi-frame extrapolation. **Why it matters / threat:** strongest systems comparator for OSS-FX disocclusion and refresh policy; exact engine MVs remain OSS’s differentiator.
- **[SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation](https://arxiv.org/abs/2608.13460)** (arXiv:2608.13460; Jisoo Jeong, Hong Cai, Jamie Menjay Lin, Hanno Ackermann, Hyeonjun Sim, Yinhao Zhu, Yunxiao Shi, Fatih Porikli; 2026-08-13; already tracked) — combines nonlinear flow candidates, confidence maps, and generative detail recovery. **Why it matters:** not deployable at OSS latency, but useful for invalid-MV, occlusion, and hole-confidence evaluation controls.[5]
- **[3DGS³: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://arxiv.org/abs/2605.11489)** remains the closest product-surface comparator: it validates joint SR+temporal rendering but retains separate SR and interpolation modules. OSS needs a shared quality/latency bake-off to substantiate the stronger single-canvas fractional-time claim.

### Inference Optimization
- **Actionable result:** do not choose a fusion/quantization architecture before #12 establishes packed state layout, #13 quantifies L2/bandwidth behavior, and #14 measures the actual VRAM footprint. MeanSR is a useful one-step training control, not a substitute for this deployment measurement chain.[3]

## Archive.org Findings
- Exact Archive.org search found **zero GS-STVSR results**; GSASR returned one unrelated 1961 CIA letter; `GRAPE` and `STSS` are heavily acronym/noise-polluted. No verified archive artifact or taken-down-paper copy surfaced.[6]
- Wayback CDX was intermittently HTTP 503 for project-specific queries, so absence of captures is inconclusive. Preserve local dependency snapshots and commit hashes rather than treating Wayback as a recovery path.

## GitHub / Open Source
- **Opportunity with licensing blocker — [SplatSuRe](https://github.com/pranav-asthana/SplatSuRe):** selective refinement in geometrically undersampled regions is a strong anti-hallucination principle for canvas spawning, but its repository has no asserted license. Reimplement the idea only after an internal ablation; do not intake code.[7]
- **Competitive ecosystem signal — [lsfg-vk](https://github.com/PancakeTAS/lsfg-vk):** active GPL Vulkan-layer frame generation demonstrates demand for Linux/presentation-path FG, but GPL plus a proprietary Lossless Scaling dependency rules out integration. Study interoperability boundaries only.[8]
- **Analysis tooling — [RE of FSR 4.1.0 Upscaling](https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling):** MIT static-analysis/pass-catalog work can inform validation methodology; it explicitly is not a runtime replacement or FSR-FG implementation.[9]
- Existing Apache-2.0 GSASR and BSD-3-Clause GaussianSR remain the cleanest reusable Gaussian-SR benchmark references. No newly pushed directly relevant open temporal-Gaussian implementation cleared the signal threshold.

## News
- **2026-08-13 — NVIDIA:** announced DLSS 4.5 adoption for upcoming RT-heavy titles and Dynamic Multi Frame Generation on RTX 50. **Implication:** title coverage and SR/FG/RT-feature packaging are competitive expectations; OSS should keep those capability claims separate.[10]
- **2026-08-10 — Intel:** announced launch-day XeSS 3 Super Resolution, Multi-frame Generation, and Xe Low Latency support for future titles across Arc A/B systems. **Implication:** OSS needs capability discovery and independent SR/FG/latency reporting on Arc rather than a monolithic upscaler toggle.[11]
- **2026-08-15 — Beast of Reincarnation patch:** added DLSS 4/4.5 and FSR 4 while explicitly deferring frame generation. **Implication:** DLL-shim/runtime design must model SR and FG as independent compatibility states.[12]
- No qualifying primary AMD, Apple/Metal, Unreal, or Unity announcement was found in the Aug. 3–17 window. Treat uncorroborated FSR 4.1/DLSS 5 reports as leads, not architecture input.

## Action Items / Follow-ups
- [ ] **Critical:** repair/restart `srcnn-v7.0-pico-005` metrics publishing/training. It has not progressed past 36,650 for 194h, and dashboard run labeling is internally inconsistent.
- [ ] Implement and measure #12 → #13 → #14 before selecting a kernel-fusion or quantization plan.
- [ ] Add an ACA-GS-style capacity-budget ablation: per-region primitive count/feature width constrained by temporal identity and L2 residency.
- [ ] Run MeanSR as a teacher-free one-step student-training control against the reparameterized HAT-distillation student; report game-VSR latency separately.
- [ ] Build OSS-FX disocclusion/refresh and invalid-MV controls from Amulet/SNM-VFI, then compare against 3DGS³ under a common quality/latency budget.
- [ ] Build DLL-shim compatibility tests with separate SR/FG status, including Vulkan/presentation path and multi-window/swap-chain lifetime cases.

## Sources

[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.04581
[3] https://arxiv.org/abs/2608.09405
[4] https://arxiv.org/abs/2608.10712
[5] https://arxiv.org/abs/2608.13460
[6] https://archive.org/advancedsearch.php?q=%22GS-STVSR%22&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&fl%5B%5D=mediatype&rows=20&page=1&output=json
[7] https://github.com/pranav-asthana/SplatSuRe
[8] https://github.com/PancakeTAS/lsfg-vk
[9] https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling
[10] https://www.nvidia.com/en-us/geforce/news/no-more-room-in-hell-2-hell-let-loose-vietnam-dlss
[11] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships
[12] https://steamdb.info/patchnotes/24741105

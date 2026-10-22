# OSS Research Brief — 2026-08-22

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`.
- **Open issues:** 12. The implementation-critical sequence remains **#12 packed low-rank canvas → #13 Nsight L2 profile → #14 VRAM measurement**, ahead of raster/bin, student-distillation, and cross-vendor work.[1]
- **README snapshot:** v7 is still the research-teacher architecture: native `(x,y,t)` Gaussians with `V_xt` covariance coupling, loss-adaptive density, and time-slice OSS-FX. The ≤1M shipping student, runtime DLL shim, and non-CUDA kernels remain unbuilt.
- **Training dashboard:** fresh JSON generated `2026-08-23T00:01:31Z` still calls `srcnn-v7.0-pico-005` “training now,” but live metrics remain **step 36,650**, total loss **0.1926767** (SR Charbonnier 0.0235991; FG Charbonnier 0.0764263; 2,304 primitives) and the trainer reports **offline/stuck for 314h**. Watcher, CF Worker, and R2 are healthy.[2][3] That is a stale-active dashboard contradiction, not new training progress.

## New Papers (arXiv)

- **Refresh caveat:** all four targeted arXiv Atom queries (Gaussian/temporal, SR, frame interpolation, and TensorRT/INT8/kernel fusion) returned HTTP 429 on this run. No paper was added from search snippets; the primary-verified entries below are preserved from the earlier same-day brief, and `BIBLIOGRAPHY.md` was not appended.

### Gaussian Splatting / Temporal
- **[Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models](https://arxiv.org/abs/2608.19556)** (arXiv:2608.19556; Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh; 2026-08-20) — adds a feed-forward 4D reconstruction reward and motion prior to limit drift/static collapse in streaming video generation. **Why it matters to OSS:** use its failure taxonomy to design long-horizon native-time-canvas checks; reject its diffusion inference path for shipping.[4]
- **[DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization](https://arxiv.org/abs/2608.02099)** (arXiv:2608.02099; Minnan Pei, Gang Li, Zeyu Zhu, Siting Wang, Junwen Si, Zhuoran Song, Yu Feng, Fangxin Liu, Xiaoyao Liang, Jian Cheng; 2026-08-03) — separates coverage/transmittance checks from blending, then compacts irregular work before parallel blending. **Why it matters to OSS:** direct dataflow input for #3 tile bin and the future fused CUDA/HIP rasterizer; already tracked in `BIBLIOGRAPHY.md`.[5]
- **[QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting](https://arxiv.org/abs/2608.18285)** (arXiv:2608.18285; Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, Kevin C. Zhou; 2026-08-18) — combines quantization-aware anchor deformation, residual coding, and change-gated densification for dynamic Gaussian streams. **Why it matters to OSS:** a concrete packing/temporal-identity control for #12, but evaluate against L2 locality before adopting streaming machinery; already tracked.[6]

### Super-Resolution / Upscaling
- **[Flow-Map Distillation on Relation Manifolds for Image Restoration](https://arxiv.org/abs/2608.05769)** (arXiv:2608.05769; Zihao He, Songhua Liu; 2026-08-06) — treats relation-based teacher/student transfer as a compositional flow map with semigroup consistency, reporting lower training variance across restoration tasks including SR. **Why it matters to OSS:** worthwhile teacher-to-≤1M-student ablation for #4/#9; validate only after a plain feature/output-distillation baseline, since it has no game-temporal or latency result.[7]
- No newly material real-time game-SR or TensorRT/INT8 result displaced the existing NanoVSR, Efficient-INT8-SISR, CDA-VSR, GRAPE, or GS-STVSR controls.

### Frame Extrapolation / Generation
- No new single-network frame-extrapolation architecture surpassed existing Amulet, ANVIL, 3DGS³, or SNM-VFI baselines. Stream4D is useful for teacher-side temporal-stability evaluation, not OSS-FX deployment.

### Inference Optimization
- No new TensorRT/INT8/kernel-fusion paper met the shipping relevance gate. Keep the RoofGS/DeGS direction provisional until #12–#14 produce actual canvas bandwidth, L2 hit-rate, and VRAM measurements.

## Archive.org Findings
- Requested Advanced Search inventories remain too noisy to be archival evidence: `gaussian splatting super-resolution` returns 3 archived GitHub-list records; `frame generation deep learning` 225 records and `real-time upscaling game` 107 records, mostly unrelated media/software. No preserved primary GS-STVSR, GSASR, GRAPE, or STSS artifact surfaced.[8]
- Treat this as a discovery failure, not evidence that any paper was removed or unavailable.

## GitHub / Open Source
- GitHub’s 14-day search found no material new open frame-generation implementation or DLSS/FSR analysis tool.
- **[SelectiveSR](https://github.com/gomzee1024/SelectiveSR)** was updated 2026-08-20 and claims scale-invariant Gaussian-SR, but has zero stars, no license, and no demonstrated temporal/game/runtime path. Monitor only; it is not a current competitor.[9]
- Newly found latent/video upscalers target generative-video pipelines rather than game rendering; no architecture or integration action for OSS.

## News
- Google News RSS shows third-party reports of fresh **FSR 4 / FSR Frame Generation** game support (Alan Wake 2) and **DLSS 4.5 / FSR 4** support (Beast of Reincarnation), plus an Intel Arc driver/XeSS watch. These are headline-level leads, not primary vendor/game-release confirmation.[10][11][12]
- No primary NVIDIA, AMD, Intel, Apple/Metal, Unreal, or Unity announcement in the 14-day scan changes OSS competitive positioning. Do not revise latency/capability claims from these reports.

## Action Items / Follow-ups
- [ ] **Critical:** restore `srcnn-v7.0-pico-005` or correct dashboard active-state copy; it is still at 36,650 and the health feed now says **314h** offline.
- [ ] Execute #12 → #13 → #14 before choosing packed attributes, quantization, sorting keys, or kernel-fusion boundaries.
- [ ] Add a small Flow-Map/semigroup-consistency distillation ablation after the baseline student pipeline exists; gate on real game-sequence temporal metrics and measured <2 ms inference.
- [ ] Deep-read DeGS for a candidate two-stage coverage/binning → compact-blend kernel experiment.
- [ ] Verify RSS FSR/DLSS/XeSS reports from primary release notes before changing compatibility or competitive claims.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://opensupersampling.com/data.json
[3] https://opensupersampling.com/status.json
[4] https://arxiv.org/abs/2608.19556
[5] https://arxiv.org/abs/2608.02099
[6] https://arxiv.org/abs/2608.18285
[7] https://arxiv.org/abs/2608.05769
[8] https://archive.org/advancedsearch.php
[9] https://api.github.com/repos/gomzee1024/SelectiveSR
[10] https://news.google.com/rss/search?q=FSR+4+when%3A14d%26hl=en-US%26gl=US%26ceid=US%3Aen
[11] https://news.google.com/rss/search?q=DLSS+4+when%3A14d%26hl=en-US%26gl=US%26ceid=US%3Aen
[12] https://news.google.com/rss/search?q=Intel+XeSS+when%3A14d%26hl=en-US%26gl=US%26ceid=US%3Aen

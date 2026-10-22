# OSS Research Brief — 2026-08-09

_Collection: local Git/README, rendered public dashboard, GitHub REST, direct arXiv pages/search, Archive.org Advanced Search + Wayback CDX, vendor primary pages, and Google News RSS. “New” means newly material to OSS; broad/off-domain hits were filtered._

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls. Recent supporting work preserves the v7 recovery snapshot, removes WSL2 DataLoader deadlock risk, enables FlashAttention 2 where available, and adds TF32/prefetch throughput work.
- **Open issues:** 10 non-PR issues visible: the immediate measurement chain is **#14** v6.2 VRAM footprint → **#13** Nsight L2 profile → **#12** packed low-rank canvas, followed by student reparameterization/distillation (#9/#4) and cross-vendor ports (#8).
- **README snapshot:** OSS remains pre-alpha. v7 is a HAT-Tiny research teacher using native `(x,y,t)` Gaussians, `V_xt` cross-covariance motion, loss-adaptive density, Mip-Splat filtering, and α=0.5 OSS-FX supervision. The ≤1M-parameter TensorRT-FP8 student and all non-CUDA runtimes remain unbuilt.
- **Training dashboard:** headline says v7 `srcnn-v7.0-pico-005` is live at **36,650 / 100,000**, loss **0.19268**, viz **36,000**; its health tooltip contradicts that, reporting **Trainer — offline, stuck at step 36,650 for 3h**. Watcher is healthy (published 1s ago), CF Worker healthy (200/99ms), and R2 origin healthy. GPU stats are unavailable. Publishing works; there is no evidence of live trainer progress. The counter reset from the prior same-day 94h display while step/loss did not advance, so treat the duration as a health-counter inconsistency, not recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- No newly submitted Gaussian-temporal paper cleared the direct OSS relevance gate. Recent, already-catalogued **ACA-GS**, **DeGS**, and **FAST-GS** remain the active density-allocation, raster-workload, and frequency-aware-motion references.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567; Nengbo Lu, Bin Zhao; 2026-05-11) — learns a per-Gaussian velocity field with global physical constraints and reports novel-view interpolation/future-frame extrapolation results. **Why it matters:** direct conceptual comparator for v7’s native time/cross-covariance motion state and a source of velocity-consistency loss ideas; already present in `BIBLIOGRAPHY.md`.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586; Nengbo Lu, Minghua Pan; 2026-05-11) — infers velocity/material factors and uses differentiable physics for long-horizon future prediction. **Why it matters:** useful only as an OSS-FX long-horizon regularization/reference; its simulation-heavy path is not compatible with a <2ms game student. Already catalogued.

### Super-Resolution / Upscaling

- **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling** (arXiv:2503.06617; Long Peng, Anran Wu, Wenbo Li, Peizhe Xia, Xueyuan Dai, Xinjie Zhang, Xin Di, Haoze Sun, Renjing Pei, Yang Wang, Yang Cao, Zheng-Jun Zha; 2025-03-09; ICLR 2026) — constructs a 2D Gaussian field from LR pixels in one pass and reports arbitrary-scale rendering in 1ms per scale. **Why it matters:** the strongest newly surfaced direct Pico-student baseline: benchmark its representation, parameter count, raster path, and arbitrary-scale quality against OSS’s current rasterizer before committing to the student head. **Opportunity:** use as a clean ablation/teacher-distillation target. **Caveat:** source code is CC-BY-NC-4.0/non-commercial, so it is research-only and cannot be incorporated into Apache-2.0 OSS.
- **SR3R**, **SplatSuRe**, **FlowGS**, and **ConFi-GS** reappeared in search but are already catalogued. Their feed-forward Gaussian fields, selective detail injection, and confidence-gated densification remain relevant ablation references, not new evidence.

### Frame Extrapolation / Generation

- No credible new single-network, exact-engine-motion-vector, real-time frame-extrapolation competitor surfaced. Existing **SPEED** and **MASS** remain appropriate offline/high-motion perceptual-teacher or benchmark comparators, not ship-path replacements for OSS-FX.

### Inference Optimization

- No material new TensorRT/INT8-SR/kernel-fusion paper surfaced. The research priority remains empirical: finish #14 → #13 → #12 and make tile occupancy, L2 hit rate, DRAM traffic, tile imbalance, and image parity acceptance gates for the packed canvas.

## Archive.org Findings

- Archive Advanced Search is noisy: `gaussian splatting super-resolution` yielded three archived GitHub/star-list snapshots; `frame generation deep learning` and `real-time upscaling game` were dominated by unrelated items. This is coverage evidence, not evidence of absence.
- Targeted Wayback CDX confirms GSASR GitHub captures from 2025-03-13, 2025-08-01/04, 2025-09-28, and 2026-06-07; GRAPE’s known arXiv page has 200 captures from 2025-02 through 2025-11; and Pixel-to-Gaussian / ContinuousSR has 200 captures in 2025. The queried GS-STVSR abstract URL returned no CDX rows. None of this is evidence of a takedown or of a material revision.

## GitHub / Open Source

- **ContinuousSR** (`peylnog/ContinuousSR`): 896 stars / 34 forks, active public code for the Pixel-to-Gaussian paper; last push 2026-01-27. High-value research comparison, but its non-commercial license prevents reuse.
- **RDNU** remains the only meaningful active open game-oriented comparison found: 15-star GPL-3.0, RDNA3-focused G-buffer-guided neural upscaler, last pushed 2026-07-24. Inspect its G-buffer contract and benchmark methodology; do not reuse GPL code.
- **bananascaler** is an MIT, zero-star GPU video-upscaler TUI last pushed 2026-08-04. Watch only: no public evidence of game integration, temporal quality, or production inference.
- **TAD-GS** has public ECCV 2026 training/evaluation code and a 2026-07-29 push; it is useful as a reproducible temporal-densification comparison, not a direct 2D/game runtime. The new MIT **Gaussian-Splat-Evaluator** detects temporal popping with RAFT flow and consecutive-frame PSNR; lift its test categories/metrics into OSS visual regression only after validating them against exact-engine-MV failure cases.
- No new open DLSS/FSR analysis tool or direct OSS competitor passed relevance review.

## News

- **NVIDIA / DLSS:** NVIDIA’s July 28 primary update is rollout news, not an architecture release: *Halo: Campaign Evolved* launches with DLSS Super Resolution + Frame Generation; *Palworld 1.0* and *Killing Floor 3* can be upgraded to DLSS 4.5 through the NVIDIA app.
- **AMD / FSR:** AMD’s primary FSR page continues to list FSR Upscaling 4.1.1 in FSR SDK 2.3, including RDNA3 discrete-GPU support; its UE plugin is still 4.0.3 pending update. Its ghosting, particle, and temporal-stability claims reinforce OSS-FX test requirements around disocclusions, particles, thin speculars, and exact-MV confidence.
- **Intel / engines:** no new primary XeSS architecture or SDK release was verified beyond XeSS SDK 3.0.2 (July 24) stability/interop fixes. A Google News RSS claim of a XeSS 3.1 UE 5.8 plugin remains an unverified lead.
- **Apple / Metal / Unity / GPU announcements:** no material primary update verified in the 7–14 day window.

## Action Items / Follow-ups

- [ ] **Critical:** restore the 3080 Ti trainer/metric emitter; require a higher step, fresh GPU telemetry, and new visualization before calling v7 training live.
- [ ] Run **ContinuousSR** as a research-only Pico baseline: compare 2D-Gaussian field construction/raster cost and quality against OSS on the same crops; retain no code or weights due to CC-BY-NC-4.0.
- [ ] Close #14 → #13 → #12 with measured bytes/frame, L2, occupancy, DRAM, tile imbalance, and image-parity gates before accepting a theoretical 32B/Gaussian layout.
- [ ] Add OSS-FX evaluation slices for moving particles, thin speculars, disocclusions, and motion-confidence failure cases; use FSR 4 claims as externally legible test classes.
- [ ] Track RDNU’s public G-buffer interface/benchmark methodology and the Intel UE-plugin lead; neither changes OSS architecture today.

## Sources

- https://opensupersampling.org
- https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
- https://arxiv.org/abs/2503.06617
- https://arxiv.org/abs/2605.10567
- https://arxiv.org/abs/2605.10586
- https://api.github.com/repos/peylnog/ContinuousSR
- https://api.github.com/repos/Adam-AtlasSoftware/RDNU
- https://web.archive.org/cdx/search/cdx?url=github.com%2FChrisDud0257%2FGSASR%2A&output=json
- https://www.nvidia.com/en-us/geforce/news/halo-campaign-evolved-mistfall-hunter-corsair-cove-dlss
- https://gpuopen.com/amd-fsr-upscaling

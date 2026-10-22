# OSS Research Brief — 2026-08-23

## Project Status Summary

- **Latest commits:** `a97a180` and `fcf8084` add tailnet-gated training controls/API and dashboard panel. The preceding sequence preserves the v7 step-5K recovery snapshot, fixes WSL DataLoader/cross-OS checkpoint resume, enables FlashAttention 2 plus TF32/prefetch, and corrects the live-PSNR proxy.
- **Open issues:** 12 open. Highest-leverage architecture work remains #12 packed low-rank canvas layout, #13 Nsight L2 profile, #9 reparameterized student backbone, #8 cross-vendor kernels, #7 FSR-4-teacher distillation, and #4 the ≤1M-parameter student.[1]
- **README snapshot:** v7 remains the research/teacher path; the ≤1M TensorRT FP8 student and cross-vendor kernels are explicitly not built. README claims OSS-FX is structurally unified but remains unmeasured as a shipped feature.
- **Training dashboard:** `srcnn-v7.0-pico-005` is still active in dashboard data at step **36,650/100,000**, total loss **0.1926767**, 2,304 primitives, and `data.json` generation time `2026-08-23T13:01:31Z`.[2] The service status simultaneously says **Trainer offline — stuck at step 36,650 for 327h**, while watcher, CF Worker, and R2 origin are healthy.[3] Treat this as a real trainer outage, not progress.

## New Papers (arXiv)

Targeted Atom discovery succeeded after the earlier same-day rate-limit failure. The material candidates below were already represented in `BIBLIOGRAPHY.md`; therefore this run makes **no duplicate bibliography append**.

### Gaussian Splatting / Temporal

- **4DAnyone: Create Anyone in 4D from a Casual Monocular Video** — Yudong Jin, Tao Xie, Qihang Zhang, Zehong Shen, Zhen Xu, Yujun Shen, Hujun Bao, Xiaowei Zhou, Yinghao Xu; 2026-08-20; arXiv:2608.20335. Uses fixed-size reference-context packing and rotating target-context routing to keep multi-view temporal reconstruction coherent as the number of views grows. **Why it matters:** the bounded-context/routing ideas are plausible training-time alternatives for long-lived OSS canvas history, but the diffusion/human-reconstruction runtime is not a shipping baseline.[4]
- **Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models** — Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh; 2026-08-20; arXiv:2608.19556. Replaces a static 3DGS consistency critic with feed-forward dynamic-4D reconstruction and a motion-magnitude prior to avoid frozen-motion solutions. **Why it matters:** its explicit anti-freeze and jitter penalties are strong candidates for OSS-FX temporal-loss ablations, independent of its diffusion backbone.[5]
- **QuARC-GS: Quantized Anchored Residual Coding for Compact Dynamic Scene Streaming with Gaussian Splatting** — Vu Trung Nghia Nguyen, Yuchen Wang, Kyung Chul Lee, Kevin C. Zhou; 2026-08-18; arXiv:2608.18285. Stores a canonical frame with compressed per-frame residuals, quantization-aware anchor deformation, and change-gated densification; reports up to 11× lower per-frame storage. **Why it matters:** directly informs #12's packed canvas representation and a thresholded primitive-update policy; deep-read its residual layout before finalizing the 32B/Gaussian claim.[6]

### Super-Resolution / Upscaling

- **MotionCraft: Latent World Modeling with Sparse Attention for Visual Upscaling** — Rong Fu, Chunlei Meng, Yangchen Zeng, Xiaowen Ma, Yongtai Liu, Wangyu Wu, Shuo Yin, Zijian Zhang, Sicheng Li, Yingrui Ji, Chenhao Wang, Simon Fong; 2026-08-09; arXiv:2608.08553. Presents a controllable VSR model combining motion fusion, sparse attention, and a compact conditional decoder. **Why it matters:** sparse non-local routing is a teacher-side efficiency/training lead only; it conflicts with the ≤1M student budget unless distilled into fixed local routing.[7]

### Frame Extrapolation / Generation

- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** — Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423. A non-neural sparse layered image-space representation with adaptive shading and explicit disocclusion handling for multi-frame extrapolation. **Why it matters:** closest practical baseline/threat to OSS-FX's “one canvas, no separate FG network” thesis; benchmark visibility/disocclusion error and marginal latency against it before making near-zero-cost positioning claims.[8]

### Inference Optimization

- No newly surfaced paper beat the existing QuARC-GS lead. Its canonical-frame-plus-quantized-residual and change-gated densification mechanisms are the concrete items worth testing for the packed canvas; avoid treating generic 4DGS reconstruction papers as runtime evidence.[6]

## Archive.org Findings

- Broad Archive.org inventory is weak: exact Gaussian-SR search returned three saved GitHub-list pages rather than paper artifacts; `"frame generation" deep learning` returned FluidFrames/DLSS Swapper builds plus an unrelated video-generation registration; `"real-time upscaling" game` returned zero results.[9]
- Targeted Wayback CDX confirms at least five distinct archived captures of the GSASR GitHub repository (beginning 2025-03-13). GS-STVSR lookup timed out, and the checked GRAPE/STSS URLs had no captures. Archive.org is therefore a fallback for known repository URLs, not a paper-discovery source.[10]

## GitHub / Open Source

- **Watch, not threat:** [`Lenovive/OSSS`](https://github.com/Lenovive/OSSS) is a new MIT Windows “window-level frame generation and upscaling” repo, created 2026-08-17 and pushed 2026-08-22, but has zero stars, forks, issues, or demonstrated benchmark/code maturity.[11]
- **Watch:** [`hy300leosquizz-ctrl/ZeroFG`](https://github.com/hy300leosquizz-ctrl/ZeroFG) is an Apache-2.0 experimental Vulkan/mobile FG POC with one star. It is adjacent to OSS's vendor-neutral/Vulkan positioning, but has no evidence yet of temporal-quality or integration maturity.[12]
- **Low signal:** [`gomzee1024/SelectiveSR`](https://github.com/gomzee1024/SelectiveSR) claims Gaussian-splatting multi-scale SR, but has zero adoption and no declared license; do not treat it as a competitor or baseline absent reproducible results.[13]
- No credible new open-source DLSS/FSR reverse-engineering or analysis tool cleared the signal threshold.

## News

- **RSS-only, unverified integration signal:** August headlines report DLSS 4.5/FSR 4 game support (including *Beast of Reincarnation* and *Phantom Blade Zero*) and alleged DLSS Ray Reconstruction 4.5 artifacts in *Modern Warfare 4* beta files.[14] No primary NVIDIA/AMD/Activision release note was retrieved, so this is not a basis for architecture claims.
- No primary-source NVIDIA, AMD, Intel/XeSS, Apple/Metal, Unreal, or Unity announcement in the last 14 days materially changed OSS positioning. The vendor/news search mostly produced low-quality or game-integration reporting rather than SDK/API changes.

## Action Items / Follow-ups

- [ ] **Ops blocker:** investigate/restart `srcnn-v7.0-pico-005`; the dashboard's current 327h trainer outage is invalidating planned convergence timing while publishing remains healthy.[3]
- [ ] **#12 / #13:** deep-read QuARC-GS, then measure canonical+residual packing, update gating, and L2 behavior against the proposed packed low-rank layout.[6]
- [ ] **OSS-FX:** run a deterministic Amulet-style layered visibility-cache baseline on disocclusion-heavy trajectories; report quality and marginal latency against the Gaussian time-slice path.[8]
- [ ] **Temporal loss ablation:** test Stream4D-style motion/jitter anti-collapse terms only after a trainer recovery, against current exact-motion-vector supervision.[5]
- [ ] Re-check OSSS and ZeroFG only if either acquires benchmarks, releases, adoption, or a real engine capture path.[11][12]

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[2] https://opensupersampling.com/data.json
[3] https://opensupersampling.com/status.json
[4] https://arxiv.org/abs/2608.20335
[5] https://arxiv.org/abs/2608.19556
[6] https://arxiv.org/abs/2608.18285
[7] https://arxiv.org/abs/2608.08553
[8] https://arxiv.org/abs/2608.10423
[9] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&rows=10&output=json
[10] https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Fgithub.com%2FChrisDud0257%2FGSASR&output=json&filter=statuscode%3A200&fl=timestamp%2Coriginal%2Cstatuscode%2Cdigest&collapse=digest
[11] https://api.github.com/repos/Lenovive/OSSS
[12] https://api.github.com/repos/hy300leosquizz-ctrl/ZeroFG
[13] https://api.github.com/repos/gomzee1024/SelectiveSR
[14] https://news.google.com/rss/search?q=DLSS+4+OR+FSR+4+OR+XeSS+when%3A14d

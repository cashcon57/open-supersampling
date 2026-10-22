# OSS Research Brief — 2026-08-21

## Project Status Summary

- **Latest commits:** `a97a180` / `fcf8084` add tailnet-gated training controls and dashboard UI. The preceding recovery/performance work preserves the v7 step-5K snapshot, fixes cross-OS resume, enables FlashAttention-2 fallback and TF32/prefetch, corrects the live-PSNR proxy, and retains FP32 smoothing under BF16.[11]
- **Open issues:** 9. Highest leverage: student distillation (#4, #7, #9), cross-vendor kernels (#8), G-buffer ablation (#10), and packed-canvas/Nsight/VRAM work (#12–#14).[11][12]
- **Training dashboard:** headline remains “live” at **36,650 / 100,000**, loss **0.19268**, viz **36K**, ETA ~1.1d. Health telemetry instead says the trainer is **offline and stuck at step 36,650 for 291h**; watcher, CF Worker, and R2 origin are healthy, while GPU stats are unavailable.[10] Compared with the previous same-day snapshot, neither step nor loss advanced; live/ETA copy is stale.[10]

## New Papers (arXiv)

### Gaussian Splatting / Temporal

- **Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models** — Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh; 2026-08-20; [arXiv:2608.19556](https://arxiv.org/abs/2608.19556). It replaces a static-3D reconstruction reward with feed-forward dynamic-4D consistency and adds scene-flow magnitude/jitter priors for long-rollout motion quality. **Why it matters to OSS:** bounded teacher-side ablation material for `V_xt` trajectory coherence and OSS-FX stability; its diffusion runtime is not a shipping-path candidate.[13]

### Super-Resolution / Upscaling

- **Binarized High-Efficiency RAW Video Restoration and Beyond (BinRVR)** — Tianyu Zhu, Ying Fu, Hesong Li, Gengchen Zhang, Xin Yuan, Yulun Zhang; 2026-08-17; [arXiv:2608.16756](https://arxiv.org/abs/2608.16756). Reports a binarized spatial-temporal restoration architecture with distribution-aware convolution and claims ~96% compute/parameter reduction at ~4% degradation in its RAW-restoration setting. **Why it matters to OSS:** low-priority quantization-training reference for the ≤1M student, not evidence that binary inference meets game-SR quality or TensorRT/FP8 constraints.[14]

### Frame Extrapolation / Generation

- No fresh direct frame-extrapolation architecture displaced **Amulet** as the deterministic OSS-FX comparator.[unverified] The direct arXiv multi-query collector was rate-limited (HTTP 429); this run verified the two surfaced candidates through their abstract pages rather than treating snippets as metadata.[13][14]

### Inference Optimization

- No new material TensorRT, INT8-SR, or kernel-fusion result cleared the architecture-impact filter. Continue to use RoofGS-style stage-level measurement—DRAM/L2 traffic, occupancy, bin/sort, raster, and composite timings—before selecting packed layout versus fusion work.[unverified]

## Archive.org Findings

- Exact `"gaussian splatting" super-resolution` returned three archived GitHub/list captures; `GS-STVSR` returned zero and `GSASR` resolved to an unrelated CIA record. `"frame generation" deep learning` and `"real-time upscaling" game` were dominated by noisy/non-paper results.[15]
- No credible removed or superseded archive copy of GS-STVSR, GSASR, GRAPE, or STSS was identified.[15] Use a targeted Wayback lookup only if a known arXiv/project URL disappears.[unverified]

## GitHub / Open Source

- **SelectiveSR** was pushed Aug 20 and claims multi-scale Gaussian-splatting SR, but has **0 stars, no license, and no demonstrated evaluation**. Watch only; it is not an adoption or competitive threat.[16]
- **ZeroFG** is an Apache-2.0 experimental Vulkan/mobile frame-generation POC (1 star; last push Aug 13). Its Vulkan interface and pacing choices can inform a future fallback path, but it has no demonstrated quality threat to OSS.[17]
- No fresh open-source DLSS/FSR analysis tool appeared in the structured search; remaining hits were generic latent-video, UI, or WIP upscalers.[16][17]

## News

- **RSS-level leads, not primary-vendor confirmation:** Aug. 19 coverage reports *Alan Wake 2* added FSR 4 and FSR Frame Generation; Aug. 10/17 coverage reports *Beast of Reincarnation* added DLSS 4.5 and FSR 4. These are integration signals, not validated SDK/API/performance changes.[18]
- RSS also surfaced an Aug. 21 report alleging FSR 4 Frame Generation frame-pacing issues. Treat it as a regression-test lead only until reproduced or vendor-confirmed; no primary NVIDIA, AMD, Intel, Apple, Unreal, or Unity upscaling release was verified in this run.[18]

## Action Items / Follow-ups

- [ ] **Ops blocker:** restore or explicitly stop/relabel Pico-005. The run is frozen at 36,650 / 0.19268 and trainer telemetry now says 291h offline.[10]
- [ ] Run a bounded Stream4D-inspired teacher-loss ablation: baseline temporal loss vs. 4D consistency plus flow magnitude/jitter penalties on high-motion/disocclusion slices; retain only if OSS-FX and held-out temporal metrics improve.[13]
- [ ] Execute #13 roofline/Nsight capture before #12 layout/fusion work; record stage-level DRAM/L2/occupancy plus tile-bin/raster cost.[12][unverified]
- [ ] Keep Amulet as the deterministic visibility/disocclusion baseline for the `t_query` OSS-FX path.[unverified]
- [ ] Monitor SelectiveSR and ZeroFG, but do not make positioning claims without reproducible metrics, active licensing, and measured runtime behavior.[16][17]

## Sources

[10] https://opensupersampling.org
[11] https://github.com/cashcon57/open-supersampling/commits/main
[12] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open
[13] https://arxiv.org/abs/2608.19556
[14] https://arxiv.org/abs/2608.16756
[15] https://archive.org/advancedsearch.php?q=%22gaussian+splatting%22+super-resolution
[16] https://github.com/gomzee1024/SelectiveSR
[17] https://github.com/hy300leosquizz-ctrl/ZeroFG
[18] https://news.google.com/rss/search?q=DLSS%204%20OR%20FSR%204%20OR%20XeSS%20when%3A14d&hl=en-US&gl=US&ceid=US:en

# OSS Research Brief — 2026-08-24

## Project Status Summary
- **Latest commit:** `b44278a test: add local Chrome Playwright dashboard smoke`; the last checked commit remains dashboard-test/infrastructure work rather than a v7-model or kernel change.
- **Open issues:** 12 open: P0 tile-bin/raster work (#1/#3), packed-canvas and measurement gates (#12–#14), student/distillation (#4/#7/#9), shader fallback (#5), launch/ablation (#6/#10), and cross-vendor ports (#8).[1]
- **README snapshot:** v7 `srcnn-v7.0-pico-005` is the active HAT-Tiny teacher; the ≤1M shipping student and non-CUDA kernels are still unbuilt.[2]
- **Training dashboard:** at `2026-08-25T00:01:02Z`, the active v7 record remains step **36,650**, canvas count **2,304**, total loss **0.19268**, SR Charbonnier **0.02360**, and FG Charbonnier **0.07643**.[4] The status endpoint calls the trainer **offline**, “stuck at step 36650 for 25h,” while watcher, Worker, and R2 are healthy.[3] The unchanged step/loss plus a growing outage counter confirms no training progress; dashboard copy still says “training now,” so the two states must not be conflated.

## New Papers (arXiv)

No new **material** arXiv paper appeared after the 2026-08-23 brief in the targeted feeds.[5][6][7] The search did recover the following already-tracked high-value papers; no bibliography append was made because their entries are already present in `BIBLIOGRAPHY.md`.

### Gaussian Splatting / Temporal
- **RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting** (Yang Luo, Yan Gong, Yongsheng Gao, Jie Zhao; 2026-08-16; arXiv:2608.15785) — compact attributes, 32-bit depth keys, bounded fast-exp, culling, dual-pixel work, and fusion target whole-pipeline 3DGS acceleration. **Why it matters to OSS:** the direct measurement template for #3/#12/#13; retain as the deep-read/reproduction priority, but validate each technique against the packed v7 canvas rather than importing its FPS claim.[5]
- **S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices** (Yiwei Li, Jiannong Cao, Weixun Gao, Rui Cao, Songye Zhu, Yinfeng Cao, Mingjin Zhang; 2026-08-20; arXiv:2608.19639) — uses octree residuals, temporal gating, and sparse updates for dynamic-GS streaming. **Why it matters:** a compact-update control for v7 residency, provided temporal primitive identity and L2 limits survive the ablation.[6]

### Super-Resolution / Upscaling
- Broad queries returned CLEAR (Gaussian sparse-view SR), arbitrary-scale GS SR work, and NanoVSR, but none is newly published since the prior brief or strong enough to change the current persistent-canvas/student plan.[5][6] Keep NanoVSR as a #9/#4 student-backbone lead; do not treat sparse-view or non-game SISR claims as game-VSR latency evidence.[2]

### Frame Extrapolation / Generation
- **Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading** (Sebastian Künzel, Fabian Schmierer, Sergej Geringer, Guido Reina, Daniel Weiskopf, Dieter Schmalstieg; 2026-08-11; arXiv:2608.10423) — a sparse, tiled, layered image-space cache models future visibility/shading and selectively refreshes tiles. **Why it matters:** already tracked, but still the best competitive control for OSS-FX disocclusion: it supports persistent representation/time-query FG while challenging pure motion warp with explicit visibility and refresh handling.[7]
- No new materially relevant single-model frame-generation or temporal-SR paper was found after yesterday’s scan.[7]

### Inference Optimization
- No new TensorRT, INT8, or kernel-fusion paper passed the deployment-relevance gate.[5] Continue treating RoofGS’s layout/key/fusion sequence as independently profiled hypotheses, not portable end-to-end performance.[5]

## Archive.org Findings
- Archive Advanced Search remains noisy: `gaussian splatting super-resolution` produced three repository/listing artifacts; `frame generation deep learning` (226 hits) and `real-time upscaling game` (107 hits) were mostly unrelated media or software snapshots. No verified primary paper artifact emerged.[11][12][13]
- Targeted Wayback CDX found no 200 capture for the GS-STVSR abstract URL (`2604.18047`). A capture exists for QuARC-GS, while Amulet had none in the check. This is archive coverage status, **not** evidence that a paper was removed or unavailable from arXiv.[14]

## GitHub / Open Source
- **IE-SRGS** (`ZheXues/IE-SRGS`) appeared today and describes “internal-external knowledge fusion” for high-fidelity 3DGS SR. It is brand-new (created/pushed 2026-08-24), has **0 stars**, no asserted license, and no independent evidence yet. **Opportunity/watchlist only:** inspect code and paper provenance before comparing it to v7 or adding it to the bibliography.[8]
- **OSSS** (`Lenovive/OSSS`) was updated today: MIT, 0 stars, Windows window-level FG/upscaling. It remains useful integration/capture plumbing reconnaissance, not a demonstrated quality competitor.[9]
- **ZeroFG** remains the only recently verified cross-platform Vulkan FG lead, but at one star and experimental-POC maturity it is a pacing/integration watch item, not a technology threat.[10]
- No verified new open-source DLSS/FSR analysis tool surfaced in the GitHub queries.[8][9][10]

## News
- Direct-primary vendor pages were not recovered for a confirmed DLSS 4, FSR 4, XeSS/Arc, Apple/Metal, NVIDIA/AMD/Intel GPU, Unreal, or Unity API announcement in the 14-day window.[15][16][17]
- Google News RSS has **headline-level leads only**: DLSS 4.5/Ray Reconstruction references in game/beta coverage; Alan Wake 2 FSR 4 support plus recurring reports of FSR FG frame-pacing problems; and Intel Arc B580 SteamOS support/XeSS-title coverage.[15][16][17] Do not promote these to OSS architecture claims until primary vendor or game release notes are checked. The reported FSR pacing issue is nevertheless a product-positioning reminder: OSS-FX needs frame-pacing and presentation telemetry alongside image metrics.

## Action Items / Follow-ups
- [ ] **P0 ops:** diagnose/restart or explain v7 trainer state at step 36,650. The healthy publication chain does not indicate training is advancing.
- [ ] **RoofGS reproduction:** profile 32-bit keys, compact attributes, bounded fast-exp, and fusion separately; use #13 Nsight data before choosing the #12 packed layout.
- [ ] **Amulet control:** prototype layered future-visibility and adaptive tile lifetime as an OSS-FX disocclusion control before claiming fractional-time rendering has a complete visibility strategy.
- [ ] **IE-SRGS triage:** verify its paper/code, license, and measured runtime before treating it as a Gaussian-SR competitor or bibliography item.
- [ ] **Presentation metric:** add frame-pacing/present-interval telemetry to OSS-FX evaluation; image quality alone will miss the FSR-style failure mode.

## Sources

[1] https://api.github.com/repos/cashcon57/open-supersampling/issues?state=open&per_page=100
[2] https://github.com/cashcon57/open-supersampling/blob/main/README.md
[3] https://opensupersampling.com/status.json
[4] https://opensupersampling.com/data.json
[5] https://arxiv.org/abs/2608.15785
[6] https://arxiv.org/abs/2608.19639
[7] https://arxiv.org/abs/2608.10423
[8] https://api.github.com/repos/ZheXues/IE-SRGS
[9] https://api.github.com/repos/Lenovive/OSSS
[10] https://api.github.com/repos/hy300leosquizz-ctrl/ZeroFG
[11] https://archive.org/advancedsearch.php?q=gaussian%20splatting%20super-resolution&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&rows=10&output=json
[12] https://archive.org/advancedsearch.php?q=frame%20generation%20deep%20learning&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&rows=10&output=json
[13] https://archive.org/advancedsearch.php?q=real-time%20upscaling%20game&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&rows=10&output=json
[14] https://web.archive.org/cdx/search/cdx?url=https%3A%2F%2Farxiv.org%2Fabs%2F2604.18047&output=json&filter=statuscode%3A200
[15] https://news.google.com/rss/search?q=DLSS+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
[16] https://news.google.com/rss/search?q=FSR+4+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen
[17] https://news.google.com/rss/search?q=Intel+XeSS+Arc+when%3A14d&hl=en-US&gl=US&ceid=US%3Aen

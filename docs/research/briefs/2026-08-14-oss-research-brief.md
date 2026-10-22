# OSS Research Brief — 2026-08-14

## Project Status Summary
- **Latest commits:** local `main` still ends at `b44278a test: add local Chrome Playwright dashboard smoke`; the requested `git log -10` contains no newer commit.
- **Open issues:** 12. The immediate performance chain remains #12 packed low-rank canvas layout → #13 Nsight L2 profile → #14 VRAM measurement. Student distillation (#4/#9), FSR 4 teacher signal (#7), vendor ports (#8), and G-buffer ablation (#10) remain open.
- **README snapshot:** v7 is the intended active research teacher: N-D `(x,y,t)` Gaussians with `V_xt` motion correlation. The ≤1M-parameter shipping CNN and all non-CUDA kernels remain unbuilt.
- **Training dashboard:** rendered dashboard now shows v7 `srcnn-v7.0-pico-005` at **36,650 / 100,000**, loss **0.19268**, but labels the trainer **offline/stuck for 122h**; GPU stats are unavailable while the watcher (published 1s ago), Cloudflare Worker, and R2 origin are healthy. This reconciles the transient v6.1/zero-step fallback seen earlier today: the v7 checkpoint is visible, but training is not progressing.[1]

## New Papers (arXiv)

Screened 45 recent results from the required Gaussian/temporal, SR, frame-generation, optimization, and game/MV queries; most were medical, reconstruction, or generic video-generation work. The material deltas below are new since the prior brief.

### Gaussian Splatting / Temporal
- **[ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields](https://arxiv.org/abs/2608.04581)** (arXiv:2608.04581; Seunghyeon Song, Joo Chan Lee, Chanung Park, Jun Young Jeong, Minseo Lee, Eunbyung Park, Jong Hwan Ko; 2026-08-05) — dynamically varies Gaussian count and feature channels per anchor according to local spatiotemporal complexity, reporting up to 1.5× better compression while retaining quality. **Why it matters to OSS:** direct architecture reference for #12’s bounded canvas: make parent/child cardinality and feature width budgeted, rather than applying a uniform primitive layout.[2]
- **[GS²CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors](https://arxiv.org/abs/2608.13502)** (arXiv:2608.13502; Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, Chi Zhang; 2026-08-13) — constrains Gaussian growth with opacity-guided splitting, mean-opacity regulation, and explicit candidate/count limits when supervision is ambiguous. **Why it matters to OSS:** not an SR system, but its bounded densification and unreliable-observation gate are a usable ablation source for disocclusions and bad engine-MV/capture inputs.[3]

### Super-Resolution / Upscaling
- No newly material real-time SR/student or TensorRT/INT8 paper displaced the already-tracked NanoVSR, QuantSR+, or DeGS leads. The new broad-query hits were out of domain or did not supply a deployable/student-relevant contribution.

### Frame Extrapolation / Generation
- **[SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation](https://arxiv.org/abs/2608.13460)** (arXiv:2608.13460; Jisoo Jeong, Hong Cai, Jamie Menjay Lin, Hanno Ackermann, Hyeonjun Sim, Yinhao Zhu, Yunxiao Shi, Fatih Porikli; 2026-08-13) — uses flow-generated intermediate candidates plus confidence maps to fuse reliable structure with generative detail around occlusions and boundaries. **Why it matters to OSS:** not viable for OSS’s latency budget or one-network thesis, but its confidence-conditioned fallback is a strong evaluation/control design for OSS-FX hole handling where engine MVs are invalid.[4]

### Inference Optimization
- No newly material TensorRT, INT8, or kernel-fusion paper found. The current action remains empirical: prototype packed state (#12), then profile L2/tile-bin behavior (#13) before selecting a quantization/kernel strategy.

## Archive.org Findings
- Archive.org advanced search returned three noisy GitHub-list snapshots for `"gaussian splatting" "super-resolution"`, three generic/deployment artifacts for `"frame generation" "deep learning"`, and zero items for `"real-time upscaling" game`; no new primary paper recovery.
- Targeted Wayback CDX confirms archived GSASR arXiv captures through 2026-06-07 and GRAPE’s project page on 2026-04-15. GS-STVSR has no captured arXiv page. The STSS CDX request returned HTTP 504, so its archive status is unverified this run.[5]

## GitHub / Open Source
- REST searches since 2026-07-31 found **no verified new game-focused Gaussian-SR repo or DLSS/FSR/XeSS analysis tool**.
- **[ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG)** is a newly created (2026-08-10), Apache-2.0 C++ Vulkan, platform-agnostic/mobile frame-generation POC. It has zero stars/forks and describes itself as experimental; monitor it as a potential open integration reference, not a competitive quality signal.[6]
- `benw46/neural-upscaler` was created today but has no description/license, zero stars, and TypeScript implementation; it is not a technical lead. Existing zero-star `bananascaler` remains a WIP TUI wrapper, not an OSS competitor.[7]

## News
- **AMD:** no 7–14-day primary FSR release was found, but AMD’s current developer guidance documents FSR Upscaling 4.1, Frame Generation 4.0.0, Ray Regeneration 1.1, and Radiance Caching preview in SDK 2.2; it claims ~1.3 ms 4K Performance-mode upscaling on RX 9070 XT and UE 5.2–5.7 plugin support. **Threat:** OSS must measure comparable full-pipeline latency, not isolated model time.[8]
- **NVIDIA:** DLSS 4.5’s announced package includes a second-generation SR transformer, Dynamic/6× Multi Frame Generation, enhanced FG, and second-generation transformer Ray Reconstruction. This remains the high-end integrated quality/feature target, not a new OSS-architecture result.[9]
- **Intel:** Intel announced launch-day XeSS 3 SR + Multi-frame Generation + Xe Low Latency for *STAR WARS: Galactic Racer* and *Tomb Raider: Legacy of Atlantis* on 2026-08-10. **Threat:** bundled SR+FG+latency is now explicit on Arc/handheld-facing hardware; OSS needs an end-to-end latency/quality capture protocol.[10]
- **Apple / Metal and engine updates:** no qualifying new primary Apple MetalFX or Unreal/Unity upscaling announcement was verified in the 14-day sweep.

## Action Items / Follow-ups
- [ ] **Critical:** restore/verify the v7 trainer on the 3080 Ti host. The dashboard has preserved the step-36,650 checkpoint but reports the trainer offline/stuck for 122h; watcher/origin health means this is trainer-side rather than publishing-side.
- [ ] Deep-read ACA-GS; turn adaptive cardinality/feature masks into a v7 #12 state-budget ablation with a temporal-identity and L2-residency acceptance gate.
- [ ] Add GS²CI-style bounded opacity/candidate regulation as a disocclusion-spawning ablation; reject it if it harms new-detail recovery from valid engine MVs.
- [ ] Use SNM-VFI’s confidence-map concept only as an OSS-FX invalid-MV/hole control; do not adopt diffusion/VFI inference.
- [ ] Clone/inspect ZeroFG only if it gains tests, benchmark evidence, or integration adoption; do not spend architecture time on a zero-signal POC.
- [ ] Establish a shared scene/capture benchmark against FSR 4.1, DLSS 4.5, and XeSS 3 covering SR, FG/extrapolation latency, invalid MVs, and disocclusions.

## Sources
[1] https://opensupersampling.com
[2] https://arxiv.org/abs/2608.04581
[3] https://arxiv.org/abs/2608.13502
[4] https://arxiv.org/abs/2608.13460
[5] https://archive.org/advancedsearch.php
[6] https://api.github.com/repos/hy300leosquizz-ctrl/ZeroFG
[7] https://api.github.com/repos/benw46/neural-upscaler
[8] https://www.amd.com/en/developer/resources/technical-articles/2026/upscale-everything-super-resolution-across-amd-hardware-.html
[9] https://www.nvidia.com/en-us/geforce/news/dlss-4-5-ray-reconstruction-1000-rtx-games-apps-out-now/
[10] https://newsroom.intel.com/client-computing/intel-gamer-days-2026-kicking-off-with-aaa-gaming-bundle-partnerships

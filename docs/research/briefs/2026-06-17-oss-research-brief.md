# OSS Research Brief — 2026-06-17

## Project Status Summary
- Latest commits:
  - `a97a180 feat: add tailnet-gated training controls to dashboard`
  - `fcf8084 feat: tailnet-gated training control API + dashboard panel`
  - `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`
  - `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`
  - `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`
  - `e1ab7fe hat.py: use flash-attn 2 directly when available, fallback to SDPA`
  - `e9b4b65 trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)`
  - `26cd9da dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB`
  - `83b4f07 v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast`
  - `baf4b02 worker: apex .com → .org 301 redirect`
- Open issues:
  - #14: D3: measure v6.2-pico-002 VRAM footprint after v6.2 wiring and checkpoint exist
  - #13: H004: run Nsight L2 hit-rate profile on 3080 Ti CUDA host
  - #12: H004: implement packed low-rank canvas layout before claiming 32B/Gaussian
  - #10: v6.2.1 ablation: decoupled G-buffer encoder before concat-fusion
  - #9: v6.2.1 ablation: RepVGG/ECBSR/RepSR reparameterization for student backbone
  - #8: v6.2: cross-vendor kernel ports (HIP / Metal / Vulkan / SYCL)
  - #7: v6.2.1: distillation pipeline with FSR 4 binary as teacher signal
  - #6: v6.2-pico-002: launch on 3080 Ti once codex T1-T5 land
  - #5: v6.2: No-ML shader variant (ships alongside v6.2)
  - #4: v6.2: HAT-Tiny student distillation (≤1M params)
- README snapshot: README now presents **v7** as the active architecture: N-D Gaussian canvas with native time axis, 3D Cholesky covariance with `V_xt` cross-correlation, parent-child loss-adaptive density, Mip-Splatting anti-aliasing filters, α=0.5 OSS-FX supervision, and a future ≤1M-param distilled student. It still correctly warns that dashboard runs are research/teacher models, not the shipping inference model.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org`. Visible state: **v7.0 Pico OSS-FX** at step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, cost **$0.76 / 12.63 GPU-hours / 5.05 kWh**, ETA ~1.2d. Operational contradiction worsened since 2026-06-16: headline and run card say `training` / `live`, but service-health tooltip says **Trainer — offline, stuck at step 30,300 for 81h**. Watcher, CF Worker, and R2 are healthy; GPU stats unavailable (`host offline or rsync failed`).

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Renderable Partial Representations for Dynamic Gaussian Splatting under Incomplete Delivery** (arXiv:2606.17212v1, 2026-06-15; Faruk Alpay, Levent Sarioglu, Yaser Hadri) — Studies dynamic Gaussian representations when only partial spatiotemporal regions/refinements have arrived, with renderable incomplete states. Why it matters to OSS: opportunity for robust degraded-mode canvas streaming/checkpoint recovery and partial Gaussian subsets; relevant to packed-canvas and dashboard/watcher failure modes, not directly SR quality.
- **TurboGS: Accelerating 3D Gaussian Splatting via Error-Guided Sparse Pixel Sampling and Optimization** (arXiv:2606.15924v1, 2026-06-14; Zheng Dong, Daifei Qiu, Pinxuan Dai, Ke Xu, Jiamin Xu) — Uses error-guided sparse pixel sampling to cut redundant 3DGS optimization work while preserving details. Why it matters to OSS: candidate teacher-training speedup and loss-driven sampling analogue for v7 parent-child density; could reduce wasted raster/loss compute on easy pixels.
- **AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancement** (arXiv:2606.17998v1, 2026-06-16; Yuhan Chen, Kunyang Huang, Fuchen Li, Zhuohan Qin, Guofa Li) — Ultra-light 2D-GS illumination-field model for fast low-light enhancement. Why it matters to OSS: weak domain match, but useful evidence that 2D Gaussian image-space fields can be made compact and fast; maybe worth mining for student-side parameterization tricks.
- **RealityBridge** (arXiv:2606.16278v1) remains relevant from yesterday’s brief; no newer primary 3DGS+SR/temporal paper displaced it in today’s scan.

### Super-Resolution / Upscaling
- **teasr: training-efficient any-step diffusion transformer for real-world image super-resolution** (arXiv:2606.16188v1, 2026-06-15; Xiang Gao, Chenxin Zhu, Yushun Fang, Qiang Hu, Xiaoyun Zhang) — Aims to make diffusion-transformer Real-ISR training efficient and any-step at inference. Why it matters to OSS: diffusion runtime is not a shipping path, but any-step/distilled teacher ideas may help generate higher-quality pseudo-targets for student distillation.
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (arXiv:2606.13580v1, 2026-06-11; Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun) — Uses event signals for VSR texture enhancement rather than only motion refinement. Why it matters to OSS: no event camera input in games, but the alignment/detail separation maps to engine motion vectors + confidence-gated detail injection.
- **SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting** (arXiv:2512.02172v2, 2025-12-01; Pranav Asthana, Alex Hanson, Allen Tu, Tom Goldstein, Matthias Zwicker) — Already in the bibliography, but official code is now active/updated on GitHub. Why it matters to OSS: useful implementation reference for selective GS super-resolution and detail consistency; not a direct real-time game upscaler.

### Frame Extrapolation / Generation
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (arXiv:2605.25909v1, 2026-05-25; Denis Gridusov, Maxim Popov, Sergey Kolyubin) — Semantic-aware dynamic 4DGS with rigid constraints and future-frame extrapolation. Why it matters to OSS: already tracked, still the strongest nearby static/rigid/transient decomposition reference for v7 native-time canvas behavior.
- **TIDES: Time-Derivative Event Simulation via Deformable Reconstruction** (arXiv:2606.02058v1, 2026-06-01; Christopher Thirgood, Dipon Kumar Ghosh, Simon Hadfield) — Continuous-time event simulation through deformable reconstruction. Why it matters to OSS: math reference for time derivatives/continuous-time supervision around α-slice rendering, but domain mismatch.
- **SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models** (arXiv:2606.00664v1, 2026-05-30; Ziheng He, Yixiang Chen, Ning Yang et al.) — Sparse keyframe interpolation to avoid generating every frame in expensive world-model rollouts. Why it matters to OSS: indirect support for keyframe/canvas reuse and frame-coherence exploitation; not a game SR/FG competitor.

### Inference Optimization
- **CREST: Deployment-Realistic Hardware-in-the-Loop NAS for Embedded Sensing Systems** (arXiv:2606.15004v1, 2026-06-12; Joseph Q. Zales, Pragya Sharma, Mani Srivastava) — Hardware-in-loop NAS for realistic deployed latency/energy rather than FLOP proxies. Why it matters to OSS: reinforces measuring actual per-backend latency for ≤1M-param student designs; avoid FLOP-only student selection.
- **Arithmetic Packing on Wide Integer Datapaths in DSP Primitives of Modern FPGA Devices** (arXiv:2606.11065v1, 2026-06-09; Titus Bornträger, Shane Fleming, Philipp Holzinger et al.) — Packs low-bit arithmetic into wide DSP datapaths. Why it matters to OSS: not GPU-focused, but relevant conceptually to INT8/FP8 packing and custom-kernel thinking.
- Today's `TensorRT`/`INT8` arXiv query was noisy and returned few SR-specific fresh items. Prior tracked `Efficient INT8 Single-Image Super-Resolution...`, `QuantSR+`, and `AutoMegaKernel` remain higher-signal for OSS student deployment.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad/noisy Archive.org hits, mostly archived GitHub/star-list/index pages; no primary GS-SR paper snapshot surfaced.
- `"frame generation" "deep learning"`: 4 hits: FluidFrames older Windows builds, DLSS Swapper Windows builds, MoCha-LD OSF registration, and a Minecraft/Radiance archive. Ecosystem breadcrumbs only.
- `"real-time upscaling" game`: 0 hits.
- Targeted broad searches: `GS-STVSR`, `GSASR gaussian splatting super resolution`, `GRAPE gaussian splatting`, and `STSS super sampling` returned 0 high-signal hits.
- Targeted Wayback CDX: `arxiv.org/abs/2605.11489` (3DGS³) has a 2026-06-07 snapshot; `arxiv.org/abs/2604.18047` (GS-STVSR) CDX timed out this run; `ShrekDino/FreeGen` and `pranav-asthana/SplatSuRe` had no GitHub CDX snapshots returned.

## GitHub / Open Source
- **pranav-asthana/SplatSuRe** — official code for SplatSuRe; default branch `splatsure`; 78 stars; pushed 2026-06-16; license `NOASSERTION`; includes renderer, metrics, setup scripts, SIBR viewers, and submodules. Why it matters: good concrete implementation to inspect for selective GS-SR consistency and evaluation; already in bibliography as paper prior art. https://github.com/pranav-asthana/SplatSuRe
- **ShrekDino/FreeGen** — Apache-2.0, 0 stars, pushed 2026-06-16; README describes “Open-Source Screen Capture → Upscaling → Frame Generation” / Vulkan / FSR 3.1, with CMake, shaders, source tree, Docker/Flatpak scaffolding. Why it matters: potential open-source competitor/integration lead, but likely immature; verify actual algorithm and whether it is wrapper/compositor vs neural upscaler. https://github.com/ShrekDino/FreeGen
- **Adam-AtlasSoftware/RDNU** — GPL-3.0, 16 stars, pushed 2026-06-09; README says RDNA3-focused recurrent G-buffer-guided neural upscaler with 8-frame history, depth/normals/albedo/flow, RCAS, and AMD WMMA target. Why it matters: still the strongest open neural upscaler competitor lead; compare its G-buffer contract and training data choices against OSS capture plan. https://github.com/Adam-AtlasSoftware/RDNU
- **hvcl/3DCT_SR_DDNM_NAB-GS** — Apache-2.0, 1 star, pushed 2026-06-12; medical 3D CT SR with diffusion projection priors and Negative Alpha Blending GS. Why it matters: domain mismatch; NAB-style reliability blending may inform splat detail injection.
- GitHub search returned no new high-signal DLSS/FSR analysis-tool repos in the last 14 days.

## News
- **NVIDIA / DLSS:** Google News RSS surfaced `CONTROL Resonant` launching September 24 with **DLSS 4.5 + path tracing**, continued DLSS 4.5 Ray Reconstruction commentary, and cross-upscaler benchmarks. NVIDIA developer blog front page did not expose fresh DLSS-specific titles in the lightweight scrape, but yesterday’s DLSS 4.5 multi-frame-generation positioning still stands. **Threat:** NVIDIA messaging is SR + RR + multi-frame-generation + UE tooling as a bundle.
- **AMD / FSR:** RSS surfaced AMD statements around **FSR 4.1** quality parity / possible RDNA3 support and continued RDNA3.5 uncertainty. **Threat/opportunity:** if AMD broadens ML FSR across RDNA3/3.5, OSS loses some “vendor-neutral ML upscaling gap” leverage; OSS needs measured unified SR+FG and openness to differentiate.
- **Intel / XeSS / Arc:** RSS surfaced Arc G3 handheld performance/driver coverage and low-signal DLSS/XeSS/FSR comparison posts. **Signal:** handheld/midrange upscaling remains visible; keep Pico/Steam Deck claims measurement-gated.
- **Apple / Metal:** RSS surfaced Apple M5 GPU benchmark/spec coverage and Mac gaming performance commentary (`Cyberpunk 2077` on Mac). **Opportunity:** Metal backend remains strategically useful, but current README correctly marks it designed-only.
- **Game engines:** No strong Unreal/Unity upscaling release-note headline surfaced in this pass. Unreal feed and Intel game-dev page returned 403 in lightweight direct fetches.

## Action Items / Follow-ups
- [ ] **Ops:** investigate v7 trainer. Dashboard has been stuck at step **30,300** for **81h** while publishing infra is healthy.
- [ ] Deep-read **Renderable Partial Representations** for robust partial-canvas rendering/recovery and how it might inform packed canvas / degraded mode.
- [ ] Deep-read **TurboGS** for error-guided sparse pixel sampling as a teacher-training speedup and as a possible v7 loss-adaptive density ablation.
- [ ] Inspect **SplatSuRe** code for selective GS-SR implementation details and metrics; compare against OSS v7 raster/detail path.
- [ ] Audit **FreeGen** for real code path, Vulkan/FSR integration, capture/compositor architecture, and whether it is OSS-adjacent competitor or wrapper shell.
- [ ] Keep **RDNU** on competitor watch; inspect RDG/G-buffer schema and whether AMD WMMA path is real/benchmarked.
- [ ] Treat DLSS 4.5 + FSR 4.1 news as positioning pressure: OSS should emphasize one-canvas SR+FG, open integration, and measured sub-2ms student path once available.

## Bibliography Update
- Appended concise entries for `Renderable Partial Representations`, `TurboGS`, and `teasr` because they add new OSS-relevant angles: partial dynamic Gaussian delivery, sparse/error-guided GS optimization, and efficient diffusion-teacher SR training.

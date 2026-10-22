# OSS Research Brief — 2026-07-10

## Project Status Summary
- Latest commits: `a97a180 feat: add tailnet-gated training controls to dashboard`; `fcf8084 feat: tailnet-gated training control API + dashboard panel`; `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`; `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`; `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`; plus flash-attn/TF32/dashboard/rasterizer fixes in the last 10 commits.
- Open issues: #14 VRAM footprint, #13 Nsight L2 hit-rate profile, #12 packed low-rank canvas layout, #10 G-buffer encoder ablation, #9 RepVGG/ECBSR/RepSR student ablation, #8 cross-vendor kernel ports, #7 FSR 4 binary distillation, #6 v6.2 launch cleanup, #5 no-ML shader variant, #4 HAT-Tiny student distillation.
- README snapshot: still describes OSS as pre-alpha, v7 architecture as the current research/teacher track, and the shipping student as not yet trained. README says v7 Pico-005 is training with native `(x,y,t)` Gaussian canvas, 3D Cholesky covariance, parent-child density, Mip-Splatting filters, and alpha=0.5 OSS-FX supervision.
- Training dashboard: rendered `opensupersampling.com` redirects to `opensupersampling.org`. Headline says `v7.0 Pico OSS-FX is training right now`, active run `srcnn-v7.0-pico-005`, step `35,200 / 100,000`, loss `0.19525`, viz at step `26,000`, 35.2% complete, ETA `~1.1d`. Service-health tooltip contradicts the live headline: `Trainer — offline`, stuck at step `35200` for `5h`; Watcher healthy published `1s` ago; CF Worker healthy HTTP 200 in `91ms`; R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Compared with the earlier same-day brief, step advanced from `30,300` to `35,200` but the trainer-health counter reset from `104h` stuck to `5h` stuck, so treat this as a dashboard/trainer health-state inconsistency rather than clean recovery.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **GaussianEmoTalker: Real-Time Emotional Talking Head Synthesis with Audio-Driven and Blendshape-Based 3D Gaussian Splatting** (arXiv:2607.00959, Haijie Yang et al., 2026-07-01) — Builds a real-time audio-driven talking-head pipeline by predicting emotion-conditioned residual deformations over GaussianBlendshapes. Why it matters to OSS: off-domain, but the neutral-to-residual-deformation framing is a useful reference for keeping a persistent Gaussian identity/canvas and only predicting cheap temporal residuals.
- **Progressive Pose-Guided 4D Animal Reconstruction from Monocular Video** (arXiv:2607.00157, Siyuan Li et al., 2026-06-30) — Uses progressive test-time optimization, temporal encoding, part anchors, and learned skinning fields for temporally consistent 4DGS animal reconstruction. Why it matters to OSS: off-domain, but progressive pose/non-rigid disentangling is a possible conceptual reference for dynamic-object/transient-canvas separation.
- **DLGStream: Dynamic Language-embedded Guassian Splatting for Open-vocabulary Enabled Free-viewpoint Video Streaming** (arXiv:2606.28840, Zhihui Ke et al., 2026-06-27) — Streams time-varying language features alongside Gaussian attributes and uses an interpolation-based deformation field to reduce temporal redundancy; claims ~43 KB average frame size and 4D frame interpolation. Why it matters to OSS: newly surfaced as material for canvas/state serialization, temporal redundancy reduction, and future thin-client/streaming variants; not directly a game SR paper.
- **Learning Efficient 4D Gaussian Representations from Monocular Videos with Flow Splatting** (arXiv:2606.29976, Shengjun Zhang et al., 2026-06-29) — Flow Splatting renders optical flow from a learned 4D Gaussian velocity field with time-varying means/covariances. Why it matters to OSS: already tracked, still the closest fresh technical analogue for comparing v7 native time-axis / `V_xt` motion encoding against explicit velocity-field supervision.
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v2, Giulio Federico et al., 2026-06-28 / revised 2026-07-01) — Replaces low-pass 2D Gaussians with Gaussian-Hermite “Brane” modes that encode high-frequency texture inside a primitive. Why it matters to OSS: direct primitive/raster ablation candidate for reducing splat count while improving edge/detail recovery; already in `BIBLIOGRAPHY.md`, no duplicate append.
- **Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution** (arXiv:2606.29400, Giulio Federico et al., 2026-06-28) — QuADA-GS predicts local complexity and adaptively allocates/densifies 2D Gaussian primitives from LR features. Why it matters to OSS: direct comparison target for v7 parent-child density and packed-canvas budgeting; already tracked.
- **DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting** (arXiv:2607.04761, Yimo Wang et al., 2026-07-06) — 4DGS semantic/geometric decoupling for surgical scenes. Why it matters to OSS: off-domain, only a weak lead for future object-aware/transient canvas channels; already tracked in 2026-07-09 bibliography.

### Super-Resolution / Upscaling
- **PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution** (arXiv:2607.06238, Lihua Wei et al., 2026-07-07) — MRI SR using dynamic resolution/SNR acquisition physics and a 2D-GS coordinate reconstruction formulation. Why it matters to OSS: off-domain; useful only as a degradation/formation-physics analogy for using engine motion/depth/G-buffer signals in training; already tracked in 2026-07-09 bibliography.
- **AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution** (arXiv:2607.00987, Geunhyuk Youk et al., 2026-07-01) — Diffusion-prior arbitrary-scale VSR with temporal consistency. Why it matters to OSS: offline teacher/pseudo-target reference, too heavy for shipping; already tracked.
- **Efficient neural supersampling on a novel gaming dataset** (arXiv:2308.01483, Antoine Mercier et al., 2023-08-03) — Game-content neural supersampling with motion vectors/depth and viewport jitter/mipmap-bias data. Why it matters to OSS: older but still a primary game-specific baseline/dataset reference; already tracked.

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, Shibo Liu, 2026-03-27 / revised 2026-04-01) — Uses codec motion vectors to prealign frames and leaves only a convolutional residual refiner, targeting mobile INT8 deployment. Why it matters to OSS: strong support for OSS’s exact-engine-motion-vector premise and a warning that spatial sampling / recurrent flow refinement are quantization-hostile; already tracked.
- **DLGStream** (arXiv:2606.28840) also belongs here as a Gaussian-state frame-interpolation/streaming lead; see Gaussian section.
- **CausalGS** / **VeloGauss** (arXiv:2605.10586 / 2605.10567) remain useful long-horizon OSS-FX regularization references, but were not new in this run.

### Inference Optimization
- No newly surfaced arXiv item in this run beats the already-tracked quantization/kernel leads. The relevant watch items remain ANVIL for motion-vector-prior INT8 interpolation, Decoder-Free Distillation / QuantSR+ for quantized image restoration, and fused INT8/FP4 kernel references from prior briefs.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad archive.org hits, all GitHub/archive-list style mirrors; no confirmed archived paper replacement for GS-STVSR/GSASR/GRAPE/STSS.
- `"frame generation" "deep learning"`: 4 broad hits; notable archived software/build pages include `dlssswapperbuildswin` and `fluidframeswin`, but no primary research artifact.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: broad acronym query is polluted (`GRAPE` dominates unrelated media/favorites); not useful as evidence.
- Wayback CDX targeted checks: `arxiv.org/abs/2604.18047` and `github.com/hongyuanyu/SPTSS` returned no 200 snapshots; GSASR and GS-STVSR GitHub CDX calls timed out. Future pass should retry exact CDX URLs with smaller per-URL calls.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — official-looking RBS code lead, pushed 2026-07-01, Python, 0 stars, no license field at scan time. Opportunity: inspect rasterizer/culling implementation once stable; Brane primitive could inform an OSS v7 detail ablation.
- **pranav-asthana/SplatSuRe** — 83 stars, Python, pushed 2026-06-16. Already known multi-view selective GS-SR baseline; still useful for selective/region-gated expensive ops.
- **Azusa309/MedGSSR** — ECCV 2026 medical GS-SR reconstruction repo, MIT, pushed 2026-06-22. Off-domain, but feed-forward hierarchical GS reconstruction may have reusable allocation patterns.
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — MIT, pushed 2026-07-02, 4 stars. Potentially important if legitimate: static structural analysis of AMD FSR 4.1.0 neural upscaler, DXIL entrypoint inventory, weight extraction, blob layout, and verification/rebuild tooling. Threat/opportunity: could accelerate competitor/reverse-engineering understanding; verify provenance before relying on it.
- **Devanshu-dev/optiscaler-client-fsr4** — MIT C# repo, 150 stars, pushed 2026-06-25; description claims OptiScaler/FSR4 mod-manager support. Competitive signal is ecosystem adoption/integration tooling rather than core model tech.
- **Spiccatobigeye413/optiscaler-client-fsr4**, **4BH15H3K-SB/optiscaler-fsr4-render-harness**, **BigBoiM7MD/pd-perfmod-fsr4** — newly pushed 2026-07-10 FSR4/OptiScaler/mod harness leads with 0 stars. Likely low-quality/spam until inspected, but they indicate active public interest around FSR4 swapping/modding.
- **ShrekDino/FreeGen** — Apache-2.0, C++, pushed 2026-06-16, describes open-source upscaling/frame generation with Vulkan + FSR 3.1. Threat only if it evolves beyond wrapper-level integration; currently low signal.
- Vendor pages checked: NVIDIA Technical Blog DLSS tag reachable; GPUOpen News reachable with many FSR/upscaling hits; Unity Blog reachable but no upscaling/Metal hits; NVIDIA consumer DLSS tag and FidelityFX SDK URL returned 404; Intel XeSS and Unreal feed returned 403.

## News
- Google News RSS was noisy/spam-prone this run; treat all non-primary items as headline-level leads.
- DLSS: one headline-level lead claims **Subnautica 2 Hotfix 3 — DLSS 4.5 & Creature Balance** (VPEsports, 2026-07-09). No primary NVIDIA/game source confirmed in this scan.
- FSR: RSS still points to broad **FSR 4 backward/older GPU support** chatter, including a previously seen CG Channel headline that AMD FSR SDK 2.3 makes FSR 4.1 work on older GPUs. Primary GPUOpen deep-read still needed.
- Intel/XeSS: RSS was mostly handheld/GPU benchmark chatter (MSI Claw/Arc G3) and low-quality mirror posts; no primary Intel source confirmed because Intel page fetch hit HTTP 403.
- Apple/Metal: AppleMagazine/MacRumors-style headlines discuss Metal gaming and Apple Silicon demand, but no concrete upscaler API change confirmed.
- Unreal/Unity: no new engine-native upscaling integration change confirmed; Unity Blog keyword scan had no DLSS/FSR/XeSS/upscaling hits.

## Action Items / Follow-ups
- [ ] Ops: dashboard contradiction needs attention — public copy says live/training with ETA, but trainer health says offline stuck at step 35200 for 5h and GPU stats are unavailable. The same-day health counter reset from 104h while step/loss advanced; verify trainer heartbeat semantics before calling the run healthy.
- [ ] Deep-read **Resonant Brane Splatting** v2 and clone/inspect `GiulioFede/ResonantBraneSplatting` for primitive expressiveness vs raster cost; possible v7 edge/detail ablation.
- [ ] Deep-read **QuADA-GS** allocation/router details against OSS parent-child density and issue #12 packed low-rank layout.
- [ ] Skim **DLGStream** for temporal redundancy / interpolation-based deformation-field ideas relevant to canvas serialization, active-mask streaming, or thin-client modes.
- [ ] Verify **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** provenance and technical depth; if real, add a competitor-analysis note for FSR 4.1 structure/weights/tooling.
- [ ] Check primary GPUOpen/FidelityFX SDK release notes for FSR 4.1 / SDK 2.3 details; RSS headlines imply a widening AMD compatibility story.
- [ ] For archive monitoring, replace broad `GRAPE`/`STSS` keyword queries with exact CDX checks against known arXiv, GitHub, and project URLs.

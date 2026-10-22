# OSS Research Brief — 2026-06-24

## Project Status Summary
- Latest commits: `a97a180` and `fcf8084` add tailnet-gated training controls/API to the dashboard; `0660f55` preserves the v7-pico-005 step-5000 snapshot + recovery doc; recent trainer/runtime work also includes WSL2 DataLoader deadlock mitigation, Linux `WindowsPath` checkpoint-resume shim, direct FlashAttention 2 fallback, TF32/prefetch, fp32 smoothing filters under bf16 autocast, and the apex `.com` → `.org` redirect.
- Open issues: #14 D3 VRAM measurement; #13 Nsight L2 profile on 3080 Ti; #12 packed low-rank canvas layout; #10/#9 v6.2.1 ablations; #8 cross-vendor kernel ports; #7 FSR 4 binary teacher-signal distillation; #6 v6.2-pico-002 launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: pre-alpha; public TL;DR still frames OSS as vendor-neutral SR + frame extrapolation via one persistent Gaussian canvas. README status says v7 `srcnn-v7.0-pico-005` is training and the shipping ≤1M-param student is not trained.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org` and shows **v7.0 Pico OSS-FX**, active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, cost **$0.76**, **12.63 GPU-hours**, **5.05 kWh**. Copy still says “v7 is training” / “live” / “updated just now,” but service health says **Trainer — offline, stuck at step 30300 for 65h**. Watcher healthy/published 0s ago; CF Worker HTTP 200 in 63ms; R2 origin HTTP 200; GPU stats unavailable. Compared with 2026-06-23, no training progress; stuck duration grew from 53h to 65h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- No genuinely new OSS-relevant GS/temporal arXiv item surfaced after yesterday’s **Lift4D** / **GaussianZoom** / **Thin-Client Interactive Gaussian Adaptive Streaming** additions. Today’s newest hit, **SignNet-1M** (arXiv:2606.24361), only matched the broad temporal/video query and is not an OSS architecture paper.
- Already-tracked recurrences: **Lift4D** (2606.23688), **Hand-4DGS** (2606.19156), **RealityBridge** (2606.16278), **ConFi-GS** (2605.24964), **FlowGS** (2605.22147), **GaussianZoom** (2605.18252), **Thin-Client Interactive Gaussian Adaptive Streaming** (2605.08699), **GS-STVSR** (2604.18047), and **Voronoi-guided Bilateral 2DGS** (2604.17727). No duplicate bibliography append for these.

### Super-Resolution / Upscaling
- No new direct game/upscaler SR paper since yesterday. Broad queries re-surfaced already-tracked **Neural Super-Resolution for Real-time Rendering with Radiance Demodulation** (arXiv:2308.06699), **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines** (2508.16024), **SlimDiffSR** (2605.02198), and older efficient/distilled SR papers.
- Practical takeaway: still no public “DLSS-class open SR+FG for games” paper that collapses SR and FG into one Gaussian state the way OSS proposes; most new SR hits are diffusion/offline or non-game remote-sensing variants.

### Frame Extrapolation / Generation
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (Songyin Wu, Deepak Vembar, Anton Sochenov, Selvakumar Panneer; 2024-05-23; arXiv:2406.18551v1) — Neural frame extrapolation for low-latency real-time rendering without G-buffer inputs. Why it matters to OSS: relevant competitor/baseline for forward frame rendering latency and artifact taxonomy; OSS’s counter-position remains exact engine motion vectors + persistent canvas instead of a separate extrapolator.
- Already-tracked recurrences: **CausalGS** (2605.10586), **VeloGauss** (2605.10567), **ReFrame** (2506.13814), **RSTT** (2203.14186). These remain relevant background but did not warrant duplicate bibliography entries.

### Inference Optimization
- No new actionable TensorRT/INT8/kernel-fusion SR paper surfaced. Queries re-surfaced **Unfolding Once is Enough** (2308.02794), **NAWQ-SR** (2212.09501), **Efficient Integer-Arithmetic-Only CNNs** (2006.11735), **ITNet** (2606.19538), and older mobile/embedded SR acceleration work.
- OSS implication: the student-deployment research queue is unchanged — prioritize measured TensorRT/FP8/INT8 path, packed canvas memory layout, and fused raster/composite kernels over adding another academic quantization baseline today.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 Archive.org hits, all broad GitHub/list snapshots; no new primary paper copy found.
- `"frame generation" "deep learning"`: 4 broad hits including FluidFrames builds, DLSS Swapper builds, and a Minecraft/Radiance archive; ecosystem artifacts only, not architecture sources.
- `"real-time upscaling" game`: 0 hits.
- Targeted terms: `GS-STVSR` 0; `GSASR` 1 unrelated CIA false positive; `GRAPE "Gaussian" "super-resolution"` 0; `STSS "super-resolution"` 0.
- Wayback CDX: `https://github.com/ChrisDud0257/GSASR` has captures from 2025-03-13, 2025-08-01, 2025-08-04. `https://github.com/W-Ted/GRAPE` returned no captures. GS-STVSR/arXiv CDX checks timed out today; inconclusive.

## GitHub / Open Source
- **Azusa309/MedGSSR** — `https://github.com/Azusa309/MedGSSR`, updated 2026-06-22, MIT, 1 star. Generalizable medical image SR 3D reconstruction via hierarchical feed-forward Gaussian splatting. Low game relevance, but hierarchy/feed-forward design may transfer to OSS density/layout thinking.
- **pranav-asthana/SplatSuRe** — `https://github.com/pranav-asthana/SplatSuRe`, updated 2026-06-16, 79 stars, official selective 3DGS SR code. Already tracked; still the strongest selective-SR/density-mask baseline lead.
- **hvcl/3DCT_SR_DDNM_NAB-GS** — `https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS`, updated 2026-06-12, Apache-2.0, 1 star. Zero-shot 3D CT SR using diffusion 2D projection priors + negative alpha blending GS; probably medical/offline only, but NAB-GS is worth a quick skim if it contains a compact splat blending trick.
- **ShrekDino/FreeGen** — `https://github.com/ShrekDino/FreeGen`, updated 2026-06-16, Apache-2.0, 0 stars. Claims Vulkan + FSR 3.1 upscaling/frame generation. Threat watch only until validated.
- **AdvaitPalve11/Jojo-Player** — `https://github.com/AdvaitPalve11/Jojo-Player`, updated 2026-06-18, MIT, 0 stars. Java/FFmpeg video player claiming upscaling + frame generation; not game-engine relevant.
- GitHub search for DLSS/FSR analysis tools and FSR4/frame-generation repos returned no relevant new repos.

## News
- News section uses Google News RSS headline-level signals plus lightweight vendor-page checks; treat non-primary headlines as leads.
- **NVIDIA / DLSS**: NVIDIA RSS headline says **Zenless Zone Zero Version 3.0** launches with **DLSS 4.5** and ray tracing. Competitive read unchanged from yesterday: NVIDIA keeps shipping transformer-SR + multi-frame-generation integrations quickly; OSS needs the “one vendor-neutral architecture, no separate FG network” message to be crisp.
- **AMD / FSR**: TechPowerUp headline says AMD explained **FSR 4.1** for RDNA 3 and promised quality parity with RDNA 4; The Verge says AMD released FSR 4.1 upscaling for older GPUs; HotHardware says Valve Proton test builds add FSR 4 for RDNA 3; PCMag says Valve is testing FSR 4 ahead of Steam Machine launch. Opportunity: Linux/Steam/older-GPU rollout keeps cross-vendor upscaling in the news, but vendor fragmentation remains a positioning opening for OSS.
- **Intel / XeSS / Arc**: PCMag headline says **Intel Arc G3 Extreme** is a strong gaming-handheld chip; TechPowerUp previously reported Arc driver 101.8826 WHQL. No major XeSS architecture shift found.
- **Apple / Metal**: no new upscaler-specific signal today. Apple Game Porting Toolkit 4 remains the only recent Metal/game-porting lead from yesterday.
- **Vendor blogs**: NVIDIA GeForce news page fetched successfully; AMD GPUOpen news page contains current FSR/FidelityFX/upscaling/frame-generation references; Intel developer graphics page returned 403. No new primary post beyond the headline-level items above.
- **Game engine updates**: no useful Unreal/Unity upscaling update surfaced in RSS today.

## Action Items / Follow-ups
- [ ] Ops: fix/restart trainer or change public copy; dashboard has been stuck at step 30,300 for 65h while still saying “live/training,” and GPU stats are unavailable.
- [ ] Deep-read **GFFE** for low-latency frame-extrapolation artifacts/metrics and use it as a competitor baseline for OSS-FX claims.
- [ ] Skim **NAB-GS** in `hvcl/3DCT_SR_DDNM_NAB-GS` only if the implementation includes a reusable alpha/blending trick; otherwise ignore as medical/offline noise.
- [ ] Keep watching **FSR 4.1 on RDNA3/Steam/Proton** as a positioning threat/opportunity for cross-vendor OSS messaging.

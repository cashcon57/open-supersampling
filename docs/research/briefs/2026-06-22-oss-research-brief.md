# OSS Research Brief — 2026-06-22

## Project Status Summary
- Latest commits: `a97a180`/`fcf8084` add tailnet-gated training controls to the dashboard; `0660f55` preserves the v7-pico-005 step-5000 snapshot + recovery doc; recent trainer/runtime work includes WSL2 `num_workers=0`, Linux `WindowsPath` checkpoint resume shim, direct FlashAttention 2 fallback, TF32/prefetch, fp32 smoothing filters under bf16 autocast, and the apex `.com` → `.org` dashboard redirect.
- Open issues: #14 D3 VRAM measurement; #13 Nsight L2 profile on 3080 Ti; #12 packed low-rank canvas layout; #10/#9 v6.2.1 ablations; #8 cross-vendor kernel ports; #7 FSR 4 binary teacher signal distillation; #6 v6.2-pico-002 launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: pre-alpha. v7 is the active teacher path: native `(x,y,t)` N-D Gaussian canvas, 3D Cholesky covariance with `V_xt`, parent-child loss-adaptive density, Mip-Splatting filters, HAT-Tiny teacher, planned ≤1M-param distilled students, and <2ms shipping target. Student models are not trained yet.
- Training dashboard: rendered state redirects to `opensupersampling.org` and shows `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**. Page copy still says “v7 is training” / “live,” but service-health tooltip says **Trainer — offline, stuck at step 30300 for 29h**. Watcher healthy/published ~1s ago; CF Worker HTTP 200 in 97ms; R2 origin HTTP 200; GPU stats unavailable. Compared with the 2026-06-21 brief, the run is still at step 30,300 and the dashboard-reported stuck duration grew from 5h to 29h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **ConFi-GS Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (Jiaxiang Li, Zongtan Zhou, Zhen Tan, Yadong Liu, Dewen Hu; 2026-05-24; arXiv:2605.24964v1) — Reliability-aware high-frequency injection for low-resolution 3DGS reconstruction. Why it matters to OSS: direct candidate for confidence-gated splat spawning/detail injection; already tracked in `BIBLIOGRAPHY.md`.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (Jiangwei Mo, Xi Lu, Hanlin Wu; 2026-05-21; arXiv:2605.22147v1) — Uses 2DGS as a continuous feature field for arbitrary-scale SR with flow-matched detail priors. Why it matters to OSS: supports the arbitrary-query 2D-Gaussian representation argument, though remote-sensing domain and generative detail priors are not the shipping runtime path; already tracked.
- **Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos** (Jeongmin Bae, Seoha Kim, Marc Pollefeys; 2026-06-17; arXiv:2606.19156v1) — Feed-forward dynamic 4DGS from egocentric video. Why it matters to OSS: useful lightweight temporal-Gaussian update reference; already tracked from the 2026-06-18 monitor.
- Noise: broad `"4D gaussian" dynamic` also returned architecture/construction and driving-sim papers with weak OSS fit; filtered out of bibliography.

### Super-Resolution / Upscaling
- **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines** (Prateek Poudel, Prashant Aryal, Kirtan Kunwar, Navin Nepal, Dinesh Baniya Kshatri; 2025-08-22; arXiv:2508.16024v3) — Predicts stationary-wavelet coefficients instead of RGB, using G-buffer attributes and temporally warped history frames for shift-invariant reconstruction. Why it matters to OSS: strong candidate ablation for student/teacher high-frequency detail heads and temporal shimmer control; added to bibliography.
- **R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision** (Weeyoung Kwon, Jeahun Sung, Minkyu Jeon, Chanho Eom, Jihyong Oh; 2025-06-19; arXiv:2506.16262v2) — Survey of 3D low-level vision under degradation, including SR/restoration around NeRF/3DGS. Why it matters to OSS: useful literature map, but not an architecture-changing primary method.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (Ce Wang, Zhenyu Hu, Wanjie Sun; 2026-05-04; arXiv:2605.02198v2) — Distills diffusion SR for lighter inference. Why it matters to OSS: teacher→student distillation pattern only; already tracked and non-game domain.

### Frame Extrapolation / Generation
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (Lufei Liu, Tor M. Aamodt; 2025-06-14; arXiv:2506.13814v1) — Caches intermediate neural features across temporally coherent real-time rendering frames for ~1.4× speedup with small quality loss. Why it matters to OSS: directly relevant to persistent canvas/student inference optimization under frame coherence; already tracked.
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (Nengbo Lu, Minghua Pan; 2026-05-11; arXiv:2605.10586v1) and **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (Nengbo Lu, Bin Zhao; 2026-05-11; arXiv:2605.10567v1) recurred in the query. Why it matters to OSS: useful long-horizon Gaussian motion priors for OSS-FX, but not new to the project.

### Inference Optimization
- **ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence** (Ashim Dhor, Rasel Mondal, Pin-Yu Chen; 2026-06-17; arXiv:2606.19538v1) — Proposes a unified learnable operator spanning conv/attention/recurrence. Why it matters to OSS: speculative kernel/operator fusion reference for future canvas-update blocks; already tracked from 2026-06-20.
- arXiv queries for `TensorRT super-resolution`, `INT8 super-resolution neural`, and `kernel fusion neural network` produced mostly older or off-domain results today. No new actionable TensorRT/INT8 paper beyond already-tracked **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** and **NAWQ-SR**.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad Archive.org hits, all GitHub/list snapshots; no new primary GS-STVSR/GRAPE/STSS paper copy found.
- `"frame generation" "deep learning"`: 4 broad hits, including DLSS Swapper builds, OSF/video-generation artifacts, and FluidFrames builds; no OSS architecture-relevant archived paper found.
- `"real-time upscaling" game`: 0 hits.
- Targeted terms: `GS-STVSR`, `GRAPE "Gaussian Rendering"`, and `STSS "super-resolution"` returned 0 useful Archive.org inventory hits; `GSASR` returned an unrelated CIA document false positive.
- Wayback targeted CDX: `github.com/ChrisDud0257/GSASR` has multiple 200 snapshots (2025-03-13, 2025-08-01, 2025-08-04). CDX requests for the GS-STVSR arXiv URL and GRAPE CVF page timed out in this run; treat as inconclusive, not absence.

## GitHub / Open Source
- **pranav-asthana/SplatSuRe** — `https://github.com/pranav-asthana/SplatSuRe`, updated 2026-06-16, 79 stars, official CVPR 2026 selective 3DGS SR code. Opportunity: selective SR/densification maps to OSS “spend detail only where needed”; already tracked.
- **hvcl/3DCT_SR_DDNM_NAB-GS** — `https://github.com/hvcl/3DCT_SR_DDNM_NAB-GS`, updated 2026-06-12, Apache-2.0, 1 star, medical CT SR via diffusion 2D projection priors + negative alpha blending GS. Low direct game relevance; alpha/composition tricks may still be useful for splat blending artifacts.
- **ShrekDino/FreeGen** — `https://github.com/ShrekDino/FreeGen`, updated 2026-06-16, Apache-2.0, 0 stars, describes Vulkan + FSR 3.1 upscaling/frame generation. Threat watch only until built/validated.
- **Adam-AtlasSoftware/RDNU** — `https://github.com/Adam-AtlasSoftware/RDNU`, updated 2026-06-09, GPL-3.0, 16 stars, RDNA3-focused G-buffer-guided neural upscaler. Most OSS-shaped open-source competitor lead; license blocks reuse, but useful to monitor for positioning/dataset choices; already tracked.
- **AdvaitPalve11/Jojo-Player** — MIT Java media player with FFmpeg, upscaling, frame generation; low OSS relevance because it is video-player, not game-engine reconstruction.
- GitHub search for new DLSS/FSR/XeSS analysis tools returned 0 relevant repos.

## News
- Web search backend was out of credits; news section uses Google News RSS headline-level signals plus direct source names from RSS, not deep primary-page extraction.
- **NVIDIA / DLSS**: NVIDIA RSS/headline signal: *CONTROL Resonant* launching Sept. 24 with **DLSS 4.5** and path tracing; more DLSS game-release velocity. Competitive note: OSS should keep the message on unified SR+FG and vendor neutrality, not branded feature cadence.
- **AMD / FSR**: TechPowerUp headline: AMD explained **FSR 4.1** for RDNA 3 and promised quality parity with RDNA 4; Wccftech headline says CachyOS Proton can download DLSS files to enable FSR 4 in more Linux games; GameGPU reports FSR 4 support in *The Outlast Trials*. Positioning hook: vendor/model fragmentation and Linux workaround complexity are useful contrasts for OSS openness.
- **Intel / XeSS / Arc**: TechPowerUp headline: Intel Arc GPU Graphics Drivers 101.8826 WHQL released. No major XeSS architecture change surfaced.
- **Apple / Metal**: AppleInsider headline: Game Porting Toolkit 4 is a major improvement for modern game coders; MacRumors covered Apple developer tooling. Relevance: keep Metal backend on the cross-vendor roadmap, but no upscaler-specific Apple signal found.
- **GPU / engine updates**: TechPowerUp “Upcoming Hardware Launches 2026” and broad GPU-market headlines appeared. Google News RSS returned no useful Unreal/Unity upscaling update for this run.

## Action Items / Follow-ups
- [ ] Ops: fix or restart trainer; dashboard has been stuck at step 30,300 for 29h while public copy still says live/training and GPU stats are unavailable.
- [ ] Deep-read **Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines** for a possible frequency/detail head or loss in the HAT teacher / ≤1M student.
- [ ] Validate **FreeGen** and **RDNU** manually before treating either as a real competitive baseline.
- [ ] Watch AMD FSR 4.1 RDNA3/RDNA4 parity claims and Linux enablement hacks as OSS vendor-neutral positioning ammo.
- [ ] Keep ReFrame-style layer/cache reuse on the student inference optimization backlog; persistent canvas + temporal feature reuse should compound.

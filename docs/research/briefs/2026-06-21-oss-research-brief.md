# OSS Research Brief — 2026-06-21

## Project Status Summary
- Latest commits: `a97a180`/`fcf8084` add tailnet-gated training controls to the dashboard; `0660f55` preserves the v7-pico-005 step-5000 snapshot + recovery doc; trainer fixes include WSL2 `num_workers=0`, cross-OS checkpoint resume, direct FlashAttention 2 fallback, TF32/prefetch, fp32 smoothing filters under bf16, and the apex `.com` → `.org` redirect.
- Open issues: #14 D3 VRAM measurement; #13 Nsight L2 profile on 3080 Ti; #12 packed low-rank canvas layout; #10/#9 v6.2.1 ablations; #8 cross-vendor kernel ports; #7 FSR 4 teacher signal distillation; #6 launch v6.2-pico-002; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: pre-alpha, v7 architecture is the active teacher path: native `(x,y,t)` Gaussian canvas, 3D Cholesky covariance with `V_xt`, parent-child loss-adaptive density, Mip-Splatting filters, HAT-Tiny teacher, ≤1M-param distilled student target, <2ms shipping budget.
- Training dashboard: rendered state shows `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**. Page copy says “v7 is training” and “live,” but service-health tooltip says **Trainer — offline, stuck at step 30300 for 5h**. Watcher healthy/published ~1s ago; CF Worker HTTP 200 in 91ms; R2 origin HTTP 200; GPU stats unavailable. This updates the earlier same-day stale-duration note; the dashboard now reports 5h stuck, not 165h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- arXiv throttled/429'd the Gaussian-splatting query in this run. No newly verified GS paper was added beyond already-tracked recent leads: **Intrinsic 4D Gaussian Segmentation** (arXiv:2606.18623), **Hand-4DGS** (arXiv:2606.19156), **GSPan** (arXiv:2606.17722), and **Envision4D** (arXiv:2606.10656). Why it matters to OSS: still the relevant cluster for object-aware canvas masks, feed-forward temporal-Gaussian updates, 2D-GS arbitrary-scale rendering, and Gaussian future prediction.

### Super-Resolution / Upscaling
- arXiv SR/VSR queries mostly hit 429/timeouts today. No new high-confidence SR paper surfaced beyond already-tracked **EvTexture++** (arXiv:2606.13580), **DTG-Restore** (arXiv:2605.30431), **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution** (arXiv:2604.21903), and **SlimDiffSR** (arXiv:2605.02198). Why it matters to OSS: scale-conditioning and teacher→student distillation remain the useful threads.

### Frame Extrapolation / Generation
- arXiv frame-generation query was 429-limited. Existing tracked references remain the useful ones: **ANVIL** (arXiv:2603.26835) for motion-vector-prior interpolation, **CausalGS** (arXiv:2605.10586), and **VeloGauss** (arXiv:2605.10567). Why it matters to OSS: ANVIL is still the closest deployment analogue for exact-engine-motion-vector OSS-FX; Gaussian velocity/causality papers inform longer-horizon regularization.

### Inference Optimization
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (Nguyen et al., 2026-04-22, arXiv:2604.20291) — Deployment-aware INT8 SR using teacher-guided training; the arXiv API result recurred today. Why it matters to OSS: direct reference for ≤1M-param INT8/FP8 student distillation and quantization-aware training; already in `BIBLIOGRAPHY.md`.
- **Towards Clip-Free Quantized Super-Resolution Networks: How to Tame Representative Images** (arXiv:2308.11365) — Older quantized SR calibration/reference-image problem. Why it matters to OSS: useful background if OSS student INT8/FP8 calibration clips high-frequency game content.
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794) — Older deployment-friendly SR transformer. Why it matters to OSS: possible teacher simplification/student block reference; already tracked.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad Archive.org hits, all GitHub/list snapshots; no new primary GS-STVSR/GRAPE/STSS copy found.
- `"frame generation" "deep learning"`: 4 broad hits, including OSF/video-project style artifacts; no OSS architecture-relevant paper found.
- `"real-time upscaling" game`: 0 hits.
- Targeted Wayback CDX: `github.com/ChrisDud0257/GSASR` still has 200 snapshots; arXiv placeholder/unknown GS-STVSR probe timed out; no evidence of a taken-down primary paper in this run.

## GitHub / Open Source
- **pranav-asthana/SplatSuRe** — updated 2026-06-16, 79 stars, official CVPR 2026 selective 3DGS SR code. Opportunity: selective SR maps to OSS “spend detail only where needed.” Already in bibliography.
- **hvcl/3DCT_SR_DDNM_NAB-GS** — updated 2026-06-12, 1 star, Apache-2.0, medical CT SR using diffusion 2D projection priors + negative alpha blending GS. Low game relevance; alpha-blending/composition tricks may be useful if OSS sees splat composition artifacts.
- **ShrekDino/FreeGen** — updated 2026-06-16, 0 stars, Apache-2.0 C++, describes open-source upscaling/frame generation via Vulkan + FSR 3.1. Threat watch only until manually built/validated.
- **Adam-AtlasSoftware/RDNU** — updated 2026-06-09, 16 stars, GPL-3.0, RDNA3-focused G-buffer-guided neural upscaler. Still the most game-upscaler-shaped open-source hit; license blocks reuse, but useful competitor/dataset monitoring. Already in bibliography.
- **AdvaitPalve11/Jojo-Player** — updated 2026-06-18, MIT Java video player with FFmpeg, upscaling, and frame generation. Low OSS impact; media-player domain, not game-engine reconstruction.
- GitHub search for `DLSS FSR XeSS analysis pushed:>=2026-06-07` returned 0 repos; no new high-confidence open-source DLSS/FSR/XeSS analysis tool surfaced.

## News
- **NVIDIA / DLSS**: Google News/RSS headline-level signal: NVIDIA says Zenless Zone Zero 3.0 adds **DLSS 4.5** + ray tracing (2026-06-16). Competitive angle remains feature/marketing velocity; OSS positioning should emphasize vendor-neutral SR+FG unification rather than trying to match every branded feature.
- **AMD / FSR**: TechPowerUp headline says AMD explained **FSR 4.1** for RDNA 3 and promises quality parity with RDNA 4 (2026-06-15); another headline reports FSR 4 support discovered in The Outlast Trials. Watch RDNA3/RDNA4/RDNA2 fragmentation as an OSS vendor-neutrality hook.
- **Intel / XeSS**: Intel Arc GPU Graphics Driver 101.8826 WHQL appeared in the 14-day window; no major XeSS architecture item surfaced.
- **Apple / Metal**: AppleInsider reports Game Porting Toolkit 4 is a big improvement for modern game coders (2026-06-17); Mac gaming/Metal path remains relevant for OSS portability, but no upscaling-specific Apple release surfaced.
- **GPU / engine updates**: TechPowerUp “Upcoming Hardware Launches 2026” appeared as a broad GPU-market headline; Google News returned no useful Unreal/Unity upscaling item in this run.

## Action Items / Follow-ups
- [ ] Ops: resolve dashboard contradiction — visible copy says “live/training” while service health says `Trainer — offline stuck at step 30300 for 5h`; GPU stats unavailable.
- [ ] Deep-read **A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution** for scale-conditioning ideas transferable to OSS SR+FX training.
- [ ] Keep **ANVIL** close during OSS-FX design; codec-MV-prior deployment logic is a strong proxy for exact-engine-MV usage.
- [ ] Validate **FreeGen** and **RDNU** manually before treating them as real competitors.
- [ ] Watch AMD FSR 4.1 RDNA3/RDNA4 model split as a positioning hook for OSS vendor-neutral transparency.
- [ ] Add retry/backoff to the research cron arXiv collector; today’s arXiv API was partially 429/timeout-limited, so absence of new arXiv papers is not a strong negative signal.

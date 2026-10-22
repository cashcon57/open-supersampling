# OSS Research Brief — 2026-06-20

## Project Status Summary
- Latest commits: `a97a180`/`fcf8084` add tailnet-gated training controls to the dashboard; `0660f55` preserves the v7-pico-005 step-5000 snapshot + recovery doc; recent trainer commits fix WSL2 DataLoader deadlock, cross-OS checkpoint resume, FlashAttention fallback, TF32/prefetch, fp32 smoothing filters, and the apex `.com` → `.org` redirect.
- Open issues: #14 D3 VRAM measurement; #13 Nsight L2 profile on 3080 Ti; #12 packed low-rank canvas layout; #10/#9 v6.2.1 ablations; #8 cross-vendor kernel ports; #7 FSR 4 teacher signal distillation; #6 launch v6.2-pico-002; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: still frames OSS as pre-alpha, unified SR + frame extrapolation via persistent 2D Gaussian canvas, HAT-Tiny teacher → ≤1M-param student, <2ms target; README text still says v6/v6.1 are active while the live dashboard has moved to v7.0 Pico OSS-FX.
- Training dashboard: visible run is `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step 26,000; page claims “v7 is training” and “live,” but service-health tooltip says **Trainer — offline, stuck at step 30300 for 153h**. Watcher published 0s ago, CF Worker returned HTTP 200 in 95ms, and R2 origin is healthy. GPU stats unavailable. This is an ops follow-up, not a research result.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos** (Bae et al., 2026-06-17, arXiv:2606.19156) — Feed-forward dynamic 3DGS from monocular egocentric video. Why it matters to OSS: useful precedent for lightweight temporal/canvas updates without per-scene optimization; already tracked in the bibliography.
- **Intrinsic 4D Gaussian Segmentation from Scene Cues** (Yazar et al., 2026-06-17, arXiv:2606.18623) — Training-free, mask-free segmentation builds a sparse affinity graph over Gaussian primitives using appearance, orientation, scale, deformation trajectory, and rendered-boundary cues. Why it matters to OSS: the same intrinsic primitive-affinity signals could drive canvas culling, region-of-interest quality boosts, or object-aware temporal stability without SAM-style auxiliary models.
- **Renderable Partial Representations for Dynamic Gaussian Splatting under Incomplete Delivery** (Alpay et al., 2026-06-15, arXiv:2606.17212) — Studies dynamic GS rendering when only partial spatiotemporal representations are available. Why it matters to OSS: relevant to robust rendering from incomplete canvas/checkpoint state and graceful degradation; already tracked.
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (Li et al., 2026-05-24, arXiv:2605.24964) — Confidence-guided detail injection for 3DGS SR. Why it matters to OSS: candidate high-frequency/detail path for Gaussian canvas SR; already tracked.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (Mo et al., 2026-05-21, arXiv:2605.22147) — Continuous-scale remote-sensing SR using flow-based GS. Why it matters to OSS: adjacent arbitrary-scale GS SR baseline; already tracked.

### Super-Resolution / Upscaling
- **MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution** (Wang et al., 2026-03-21, arXiv:2603.20690) — Distills a flow/diffusion SR teacher into a one-step student by learning average PF-ODE velocity, with optional few-step refinement. Why it matters to OSS: strong teacher→student pattern for compressing expensive restoration priors; worth comparing against HAT-teacher distillation for the ≤1M-param student.
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (Nguyen et al., 2026-04-22, arXiv:2604.20291) — Deployment-aware INT8 SISR with teacher-guided training. Why it matters to OSS: directly relevant to the shipping INT8/FP8 student and already tracked.
- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (Lv et al., 2026-03-23, arXiv:2603.22271) — One-step VSR distillation from diffusion-style teachers. Why it matters to OSS: temporal teacher-student training pattern for real-time video/game SR; already tracked.
- **SlimDiffSR: Toward Lightweight and Efficient Remote Sensing Image Super-Resolution via Diffusion Model Distillation** (Wang et al., 2026-05-04, arXiv:2605.02198) — Lightweight diffusion-distilled SR in remote sensing. Why it matters to OSS: domain is different, but distillation mechanics remain relevant; already tracked.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (Lu & Pan, 2026-05-11, arXiv:2605.10586) — Learns velocity/material dynamics from video and uses differentiable physics for long-term future frame extrapolation. Why it matters to OSS: not a game upscaler, but it reinforces that Gaussian state can support future-frame prediction; possible reference for longer-horizon OSS-FX beyond single-frame alpha interpolation.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (Lu & Zhao, 2026-05-11, arXiv:2605.10567) — Models Gaussian velocity fields from dynamic videos. Why it matters to OSS: conceptually aligned with native time/velocity axes in v7; already tracked.
- **ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering** (Liu & Aamodt, 2025-06-14, arXiv:2506.13814) — Caches layers for temporally coherent real-time rendering networks. Why it matters to OSS: possible latency reduction if student/canvas updates are separable across frames; already tracked.
- **GFFE: G-buffer Free Frame Extrapolation for Low-latency Real-time Rendering** (Wu et al., 2024, arXiv:2406.18551) — Frame extrapolation without G-buffers. Why it matters to OSS: competitor/contrast point; OSS’s exact engine-MV path should be better positioned when vectors are available.

### Inference Optimization
- **ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence** (Dhor et al., 2026-06-17, arXiv:2606.19538) — A unified learned-kernel operator with tiled kernel fusion, Monte Carlo integration, and low-rank factorization. Why it matters to OSS: speculative but interesting for v8+ if the canvas update can share one fused operator across convolution/attention/recurrent temporal roles; not a near-term shipping dependency.
- **Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers** (Maczan, 2026-02-09, arXiv:2604.02344) — Measures WebGPU dispatch overhead across vendors/backends. Why it matters to OSS: relevant to cross-vendor backend strategy if OSS ever targets WebGPU/browser-like runtimes; already tracked.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 Archive.org hits, all broad GitHub/list snapshots rather than primary papers; no new GS-STVSR/GSASR/GRAPE/STSS archival copy identified.
- `"frame generation" "deep learning"`: 4 hits, including OSF MoCha-LD and archived FluidFrames/DLSS Swapper builds; no architecture-relevant OSS prior art found.
- `"real-time upscaling" game`: 0 hits.
- Broad `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16k+ noisy hits because “grape” is overloaded; not useful without targeted URL CDX checks.

## GitHub / Open Source
- **pranav-asthana/SplatSuRe** — updated 2026-06-16, 78 stars, official CVPR 2026 code for selective multi-view-consistent 3DGS super-resolution. Opportunity: selective SR principle maps cleanly to OSS quality boosts only where the canvas needs detail.
- **hvcl/3DCT_SR_DDNM_NAB-GS** — updated 2026-06-12, Apache-2.0, zero-shot 3D CT SR using diffusion 2D projection priors + negative alpha blending GS. Low direct game relevance, but NAB-GS may be worth a skim if alpha composition artifacts appear.
- **ShrekDino/FreeGen** — updated 2026-06-16, Apache-2.0 C++, “open-source upscaling and frame generation — Vulkan, FSR 3.1.” Threat/positioning watch: if real, it is adjacent open-source FG/upscaling infrastructure; currently 0 stars, needs validation before treating as competitor.
- **Adam-AtlasSoftware/RDNU** — updated 2026-06-09, GPL-3.0, RDNA3-focused ML upscaler using decoupled G-buffer guidance, trained/fine-tuned on GameIR/M3VIR/vKITTI2/Sintel. Threat/opportunity: more directly game-upscaler-shaped than most hits; license is incompatible for code reuse, but architecture/dataset choices are worth monitoring.
- No new GitHub hits for DLSS/FSR analysis tools or XeSS open-source upscalers in the 14-day window.

## News
- **NVIDIA / DLSS 4.5**: Google News/RSS shows NVIDIA game-launch pushes around DLSS 4.5 + path tracing (e.g. CONTROL Resonant, Zenless Zone Zero). Competitive signal: NVIDIA continues bundling upscaling with RT/PT game launches, so OSS positioning should emphasize vendor-neutral + one-architecture SR/FG rather than raw feature count.
- **AMD / FSR 4.1**: AMD headline “Upscale Everything: Super-Resolution Across AMD Hardware” plus press coverage says RDNA 3 gets a different FSR 4.1 model, with RDNA 2 delayed by optimization needs. Opportunity: cross-generation model fragmentation is exactly where OSS can argue for transparent/vendor-neutral kernels.
- **Intel / XeSS**: no recent RSS hits from the configured query.
- **Apple / Metal**: WWDC-era developer tooling/Game Porting Toolkit 4 coverage continues; Cyberpunk-on-Mac 60 FPS articles suggest Metal/game-porting performance remains an active narrative. Relevance: Metal backend remains strategically important for vendor-neutral claims.
- **Game engines**: Unreal Engine 5 game performance/upscaler benchmark headlines continue (Gothic 1 Remake DLSS/FSR/XeSS/TSR comparisons). No major Unity upscaling item found in the query window.

## Action Items / Follow-ups
- [ ] Ops: investigate dashboard contradiction — “live/training” copy vs `Trainer — offline stuck at step 30300 for 153h`, GPU stats unavailable.
- [ ] Deep-read **Intrinsic 4D Gaussian Segmentation** for object/primitive affinity signals usable in canvas culling, stability masks, or adaptive quality.
- [ ] Deep-read **MFSR** and compare MeanFlow one-step distillation against current HAT-Tiny teacher → ≤1M student plan.
- [ ] Skim **CausalGS** for longer-horizon Gaussian future prediction; likely v8+ only, but it may inform native time-axis regularization.
- [ ] Validate **RDNU** and **FreeGen** repos manually before citing as competitors; if real, add to the open-source competitor watchlist.
- [ ] Watch AMD FSR 4.1 RDNA3/RDNA2 split as a positioning example for OSS cross-vendor transparency.

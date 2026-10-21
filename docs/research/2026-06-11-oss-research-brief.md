# OSS Research Brief — 2026-06-11

## Project Status Summary
- Latest commits:
  - `a97a180` feat: add tailnet-gated training controls to dashboard
  - `fcf8084` feat: tailnet-gated training control API + dashboard panel
  - `0660f55` wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc
  - `56a9a65` trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)
  - `d1255c8` trainer: WindowsPath shim on Linux for cross-OS checkpoint resume
  - `e1ab7fe` hat.py: use flash-attn 2 directly when available, fallback to SDPA
  - `e9b4b65` trainer: TF32 conv + DataLoader prefetch (quality-neutral throughput)
  - `26cd9da` dashboard: fix Live PSNR proxy to L1→RMSE-corrected dB
  - `83b4f07` v7 rasterizer: keep smoothing filters in fp32 under bf16 autocast
  - `baf4b02` worker: apex .com → .org 301 redirect
- Open issues: 12 open. Top visible: #14 D3 VRAM footprint, #13 Nsight L2 hit-rate profile, #12 packed low-rank canvas layout, #10 decoupled G-buffer encoder ablation, #9 RepVGG/ECBSR/RepSR student backbone ablation, #8 cross-vendor kernel ports, #7 FSR4-binary distillation teacher, #6 v6.2-pico-002 launch, #5 no-ML shader variant, #4 HAT-Tiny student distillation.
- README status snapshot: README now positions **v7** as current research/teacher run (`srcnn-v7.0-pico-005`), with N-D Gaussian canvas + native time axis, 3D Cholesky covariance with V_xt motion correlation, parent-child loss-adaptive density, Mip-Splatting AA filters, and alpha=0.5 OSS-FX supervision. Shipping student remains untrained.
- Training dashboard: `https://opensupersampling.com` redirects to `.org`. Browser-rendered health showed **Trainer offline/stuck at step 20950 for ~70h**, Watcher healthy/published 1s ago, CF Worker HTTP 200 in 78ms, R2 origin HTTP 200. Dashboard text appears partially stale/conflicting (`v6.1 Pico` / `srcnn-v6.2-pico-002` visible while README says v7-pico-005); treat trainer status as the notable operational finding.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving** (arXiv:2606.10656, 2026-06-09) — Feed-forward 4D-GS future extrapolation with future pose prediction, in-layer temporal attention, and conditioned motion lifting. Why it matters to OSS: external validation that Gaussian primitives are becoming a future-frame prediction substrate; compare its learned pose/motion handling against OSS's stronger exact engine-motion-vector prior. Already in bibliography from 2026-06-09.
- **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting** (arXiv:2606.03479, 2026-06-02) — Couples differentiable rigid-body simulation with object-wise Gaussians to preserve occluded dynamic objects when photometric gradients vanish. Why it matters to OSS: useful design prior for disocclusion/object permanence in a persistent canvas, especially when engine depth/MVs expose occlusion transitions.
- **R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction** (arXiv:2605.25909, 2026-05-25) — Adds semantic identity encodings and rigid-body constraints to dynamic 4D-GS for efficient future extrapolation. Why it matters to OSS: candidate idea for separating static/rigid/transient canvas subsets and stabilizing extrapolation under object motion.
- **RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video** (arXiv:2605.23672, 2026-05-22) — Uses static, rigid, and transient Gaussian primitives to capture multiple temporal motion scales. Why it matters to OSS: reinforces the v7 need for persistent vs transient primitive classes; already in bibliography.

### Super-Resolution / Upscaling
- **ConFi-GS: Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964, 2026-05-24) — Reliability-aware high-frequency injection for GS-SR from low-resolution multi-view images. Why it matters to OSS: strong candidate for confidence-gated detail spawning/pseudo-label filtering; already in bibliography.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (arXiv:2605.22147, 2026-05-21) — Flow-matching detail priors on top of 2D-GS for arbitrary-scale remote-sensing SR. Why it matters to OSS: supports continuous-scale 2D-GS SR direction, but diffusion/flow component is likely too slow for shipping path; already in bibliography.
- **DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution** (arXiv:2605.13182, 2026-05-13) — One-step diffusion STVSR that processes whole videos and fuses spatial/temporal conditioning. Why it matters to OSS: teacher-side reference for coupled SR+VFI training; not a real-time student architecture. Already in bibliography.

### Frame Extrapolation / Generation
- **3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering** (arXiv:2605.11489, 2026-05-12) — Post-render framework for 3DGS super sampling + frame interpolation using gradient-aware super sampling and GRU refinement. Why it matters to OSS: closest direct competitive paper to OSS's unified SR+FG positioning, though it is post-render 3DGS rendering rather than game-engine input reconstruction. Already in bibliography.
- **TIDES: Time-Derivative Event Simulation via Deformable Reconstruction** (arXiv:2606.02058, 2026-06-01) — Continuous-time event-camera simulator built on dynamic Gaussian splatting. Why it matters to OSS: marginal but interesting for continuous-time raster/query math; could inform alpha-time derivative supervision, not a core game-upscaling threat.

### Inference Optimization
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291, 2026-04-22) — Compact extract-refine-upsample SISR graph with lightweight reparameterizable backbone, deployment-aware INT8 quantization, teacher-guided training, DCT-domain supervision, and confidence-weighted losses. Why it matters to OSS: directly relevant to ≤1M-param student distillation and INT8/FP8 calibration; worth deep-reading for student loss staging.
- **Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution** (arXiv:2308.02794) — SR transformer block designed to avoid deployment-unfriendly redundant ops for TensorRT. Why it matters to OSS: possible teacher simplification / distillation reference; already captured in the 2026-06-10 bibliography additions.
- **Compressed-Domain-Aware Online Video Super-Resolution** (arXiv:2603.07694) — Uses codec motion vectors, residual maps, and frame types to reduce alignment cost in online VSR. Why it matters to OSS: strong prior for exact-engine-MV reuse and gating; already in bibliography.

## Archive.org Findings
- Generic `web_search` backend failed with Firecrawl payment/credit exhaustion, so the run used Archive.org structured APIs.
- Archive.org `advancedsearch` for `"gaussian splatting" AND "super-resolution"` and `"frame generation" AND "deep learning"` returned placeholder/empty-doc hits; `"real-time upscaling" AND game` returned no useful docs.
- Wayback CDX checks:
  - `https://github.com/ChrisDud0257/GSASR` has archived 200 snapshots from 2025-03-13, 2025-08-01, and 2025-08-04.
  - `https://arxiv.org/abs/2604.18047` returned no CDX hits in this run.
  - `https://github.com/davelailai/GRAPE` CDX timed out.
  - `https://github.com/tljxyys/GaussianSR` returned no CDX hits.
- Follow-up: rerun Wayback CDX for GS-STVSR/GRAPE/GaussianSR with smaller per-URL calls; current run does not prove absence.

## GitHub / Open Source
- `hvcl/3DCT_SR_DDNM_NAB-GS` — zero-shot 3D CT SR using diffusion-based 2D projection priors + Negative Alpha Blending Gaussian Splatting; updated 2026-06-12, 0 stars. Low OSS threat due medical/CT domain, but another data point for GS as SR representation.
- `Adam-AtlasSoftware/RDNU` — **Radeon Decoupled Neural Upscaler**, RDNA3-focused ML upscaler using decoupled G-buffer guidance, pretrained/fine-tuned on M3VIR/GameIR/vKITTI2/MPI-Sintel; updated 2026-06-09, 15 stars. This is the most relevant open-source monitor hit: potential competitor/opportunity around AMD-targeted G-buffer-guided upscaling.
- `shepherdscientific/neural-spatial-upscaler` — ternary CNN FPGA upscaler for 240p→1080p; updated 2026-06-02. Low direct relevance, but reinforces interest in ultra-low-bit/upscaler hardware paths.
- `nova-studios-hw/novagpu-ts1t` — open-source GPU RTL mentioning hardware ray tracing/frame generation; not directly OSS-useful today.
- No useful new DLSS/FSR analysis-tool repos found via GitHub Search API (`DLSS FSR analysis tool pushed:>=2026-05-28` returned empty).

## News
- **NVIDIA / DLSS:** Google News RSS shows NVIDIA announced **DLSS 4.5 Ray Reconstruction** / second-gen transformer-based Ray Reconstruction around 2026-05-31 / 2026-06-01, with press coverage continuing through 2026-06-08. Threat: NVIDIA is pushing transformer reconstruction quality and ray reconstruction messaging hard; OSS should keep the teacher/student transformer narrative explicit and avoid overclaiming until v7 metrics recover.
- **AMD / FSR:** RSS shows FSR 4 passed a claimed 300+ game milestone and that FSR 4.1 support for RDNA 3.5 APUs is disputed/doubtful in press reports. Opportunity: vendor-neutral OSS can position against AMD feature segmentation if the open student can run broadly.
- **Intel / XeSS:** RSS shows XeSS 3 Multi Frame Generation coverage (Death Stranding 2 patch) plus Intel Arc driver updates / handheld Arc G-series coverage. Threat: Intel is moving MFG into shipped games; OSS-FX needs a concrete demo path to stay credible.
- **Apple / Metal:** No primary Apple Metal upscaling announcement surfaced; mostly benchmark/game-port chatter (Cyberpunk on Mac, M4/M5 GPU specs). No immediate action.
- **Engine updates:** Google News RSS query for Unreal/Unity upscaling returned no useful recent items in this run.

## Action Items / Follow-ups
- [ ] **Operational:** investigate dashboard/trainer mismatch: browser shows trainer stuck at step 20950 for ~70h and dashboard text conflicts with README's v7-pico-005 status.
- [ ] Deep-read **Efficient INT8 Single-Image SR via Deployment-Aware Quantization and Teacher-Guided Training** for OSS student distillation staging, DCT loss, and confidence weighting.
- [ ] Deep-read **R5DGS** and **PersistGS** for rigid/object-permanence ideas applicable to v7 persistent vs transient canvas handling.
- [ ] Inspect `Adam-AtlasSoftware/RDNU` README/code to judge whether Radeon/G-buffer-guided neural upscaling is a real competitor or just a stub.
- [ ] Rerun Archive.org CDX for GRAPE / GS-STVSR / GaussianSR with retries; this run hit timeout/empty results.
- [ ] Watch DLSS 4.5 Ray Reconstruction and XeSS 3 MFG rollout as competitor positioning pressure on OSS-FX.

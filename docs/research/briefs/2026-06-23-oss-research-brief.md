# OSS Research Brief — 2026-06-23

## Project Status Summary
- Latest commits: `a97a180` and `fcf8084` add tailnet-gated training controls/API to the dashboard; `0660f55` preserves the v7-pico-005 step-5000 snapshot + recovery doc; recent trainer/runtime work includes WSL2 DataLoader deadlock mitigation, Linux `WindowsPath` checkpoint-resume shim, direct FlashAttention 2 fallback, TF32/prefetch, fp32 smoothing filters under bf16 autocast, and the apex `.com` → `.org` redirect.
- Open issues: #14 D3 VRAM measurement; #13 Nsight L2 profile on 3080 Ti; #12 packed low-rank canvas layout; #10/#9 v6.2.1 ablations; #8 cross-vendor kernel ports; #7 FSR 4 binary teacher-signal distillation; #6 v6.2-pico-002 launch; #5 no-ML shader variant; #4 HAT-Tiny student distillation.
- README snapshot: still pre-alpha; public TL;DR describes one vendor-neutral SR + frame-extrapolation architecture via persistent 2D Gaussian canvas and exact engine motion vectors. README status text is partly stale/v6-oriented, but the dashboard headline now says **v7.0 Pico OSS-FX**.
- Training dashboard: rendered `https://opensupersampling.com` redirects to `.org` and shows **v7.0 Pico OSS-FX**, active run `srcnn-v7.0-pico-005`, step **30,300 / 100,000**, loss **0.18269**, viz @ step **26,000**, cost **$0.76**, **12.63 GPU-hours**, **5.05 kWh**. Public copy still says “v7 is training” / “live” and “updated just now,” but service health says **Trainer — offline, stuck at step 30300 for 53h**. Watcher healthy/published 1s ago; CF Worker HTTP 200 in 74ms; R2 origin HTTP 200; GPU stats unavailable. Compared with the 2026-06-22 brief, no training progress; stuck duration grew from 29h to 53h.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild** (Yehonathan Litman, Xiaoxuan Ma, Manan Shah, Nicolas Ugrinovic, Kris Kitani, Fernando De la Torre, Shubham Tulsiani; 2026-06-22; arXiv:2606.23688v1) — Uses temporally consistent single-view 3D priors plus occlusion-aware optimization to initialize/refine deformable 3DGS from monocular video. Why it matters to OSS: good disocclusion/occlusion-recovery prior for persistent Gaussian state, but test-time optimization + diffusion completion is teacher/offline only, not shipping runtime.
- **GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting with Geometric and Semantic Guidance** (Jiale Shi, Jiarui Hu, Zesong Yang, Kaixuan Luan, Hujun Bao, Zhaopeng Cui; 2026-05-18; arXiv:2605.18252v1) — Builds extreme zoom-in rendering with multi-view-consistent SR, depth-based feature warping, VLM-driven detail synthesis, and expandable continuous LOD for 3DGS. Why it matters to OSS: LOD/visibility modulation and anti-aliased cross-scale rendering are relevant to canvas density/quality levels; generative detail path is likely teacher-only.
- **Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3** (Emanuele Artioli, Philipp Fößl, Daniele Lorenzi, Farzad Tashtarian, Mahdi Dolati, Cheng-Hsin Hsu; 2026-05-09; arXiv:2605.08699v1) — Streams view-dependent 2D projections from backend-rendered 3DGS over QUIC/HTTP3 to lightweight clients. Why it matters to OSS: not SR, but relevant if OSS ever exports Gaussian canvas/history to cloud or thin-client pipelines; bandwidth/latency framing may inform canvas serialization.
- Already-tracked recurrences: **Hand-4DGS** (arXiv:2606.19156), **Intrinsic 4D Gaussian Segmentation** (2606.18623), **GSPan** (2606.17722), **RealityBridge** (2606.16278), **Envision4D** (2606.10656), **PersistGS** (2606.03479), **ConFi-GS** (2605.24964), **FlowGS** (2605.22147), and **GS-STVSR** (2604.18047). No duplicate bibliography append for these.
- Filtered noise: construction documentation, head-editing, and human/avatar diffusion papers mention temporal 3DGS but have weak direct fit to OSS runtime architecture.

### Super-Resolution / Upscaling
- **DTG-Restore: Training-Free Diffusion Refinement for Generative Video Super-Resolution** (Hidir Yesiltepe, Koutilya PNVR, Gaurav Pathak, Navaneeth Bodla, Bharat Singh, Pinar Yanardag, Jinrong Xie; 2026-05-28; arXiv:2605.30431v1) — Training-free diffusion refinement for distorted/low-res video that decouples unconditional guidance timesteps to improve geometry and temporal stability. Why it matters to OSS: possible offline teacher/pseudo-target cleanup source; too slow/stochastic for the ≤1M-param real-time student.
- **EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution** (Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun; 2026-06-11; arXiv:2606.13580v1) recurred. Why it matters to OSS: texture branch + temporal texture alignment remain useful analogies for detail losses, but already tracked.
- **SwiftI2V** and **BurstGP** appeared in broad VSR/generation queries, but are image-to-video/burst-photo generative systems rather than game upscalers; no bibliography action.

### Frame Extrapolation / Generation
- No new direct “single model SR+FG for games” paper surfaced beyond already-tracked **Envision4D**, **CausalGS**, and **VeloGauss**. The field continues to move toward Gaussian-state future prediction, but most new hits rely on learned/diffusion priors rather than exact engine motion vectors.
- Competitive read: OSS’s strongest differentiator remains using one persistent canvas with deterministic engine-MV alpha rendering, avoiding a second frame-generation network.

### Inference Optimization
- **ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence** (Ashim Dhor, Rasel Mondal, Pin-Yu Chen; 2026-06-17; arXiv:2606.19538v2) recurred under kernel-fusion search. Why it matters to OSS: speculative fused-operator reference for canvas-update kernels, already tracked.
- No new actionable TensorRT/INT8 SR deployment paper surfaced today beyond already-tracked **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training**, **QuantSR+**, and **NAWQ-SR**.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad Archive.org hits, all GitHub/list snapshots; no new primary paper copy found.
- `"frame generation" "deep learning"`: 3 broad hits including DLSS Swapper builds, FluidFrames builds, and a Minecraft/Radiance archive; useful as ecosystem artifacts only, not OSS architecture sources.
- `"real-time upscaling" game`: 0 hits.
- Targeted terms: `GS-STVSR` returned 0; `GSASR` returned an unrelated CIA false positive; `GRAPE` and `STSS` were heavily polluted by unrelated acronym/common-word results. No new reliable archived GS-STVSR/GSASR/GRAPE/STSS source found.
- Wayback CDX checks for GS-STVSR arXiv, GSASR GitHub, GS-STVSR GitHub, and GRAPE GitHub timed out; inconclusive, not evidence of absence.

## GitHub / Open Source
- **Azusa309/MedGSSR** — `https://github.com/Azusa309/MedGSSR`, updated 2026-06-22, MIT, 1 star. Generalizable medical image SR 3D reconstruction via hierarchical feed-forward Gaussian splatting. Low game relevance, but hierarchy/feed-forward design may be worth a quick skim if code is complete.
- **pranav-asthana/SplatSuRe** — `https://github.com/pranav-asthana/SplatSuRe`, updated 2026-06-16, 79 stars, official selective 3DGS SR code. Already tracked; still relevant as a “spend SR only where needed” baseline for density/detail maps.
- **ShrekDino/FreeGen** — `https://github.com/ShrekDino/FreeGen`, updated 2026-06-16, Apache-2.0, 0 stars. Claims Vulkan + FSR 3.1 upscaling/frame generation. Threat watch only until validated.
- **Adam-AtlasSoftware/RDNU** — `https://github.com/Adam-AtlasSoftware/RDNU`, updated 2026-06-09, GPL-3.0, 16 stars. RDNA3-focused G-buffer-guided neural upscaler; still the most OSS-shaped open-source competitor lead, but license blocks reuse.
- GitHub search for new DLSS/FSR analysis tools returned no relevant repos.

## News
- News section uses Google News RSS headline-level signals plus lightweight vendor-page checks; not all items were primary-source deep-read.
- **NVIDIA / DLSS**: primary NVIDIA GeForce pages confirm **DLSS 4.5**: a 2nd-gen transformer Super Resolution model available to all RTX GPUs through NVIDIA app, **Dynamic Multi Frame Generation** for RTX 50, and up to **6X Multi Frame Generation** / five generated frames per rendered frame. NVIDIA explicitly claims 5× compute vs the first transformer SR model and linear-space training/inference for lighting and temporal stability. Competitive threat: NVIDIA is pushing both SR quality and adaptive multi-frame-generation UX fast; OSS needs crisp “one architecture, vendor-neutral, no second FG network” positioning and should watch the 5×-compute-vs-quality trade.
- **AMD / FSR**: TechPowerUp headline says AMD explained **FSR 4.1** for RDNA 3 and promised quality parity with RDNA 4; The Verge headline says AMD released FSR 4.1 upscaling for older graphics cards; HotHardware/Wccftech headlines show Proton/Linux paths for FSR 4/RDNA3. Opportunity: vendor fragmentation + Linux workaround complexity remain strong OSS positioning ammo.
- **Intel / XeSS / Arc**: TechPowerUp headline: Intel Arc GPU Graphics Drivers 101.8826 WHQL released. No major XeSS architecture shift found.
- **Apple / Metal**: AppleInsider headline says Game Porting Toolkit 4 is a big improvement for modern game coders. Keep Metal backend on the roadmap; no Apple upscaler-specific signal surfaced.
- **Vendor blogs**: NVIDIA blog front page was mostly MLPerf/agentic/CUDA runtime content; AMD GPUOpen page exposed FSR SDK/upscaling/frame-generation sections but no clearly new primary post in the fetched headings; Intel developer page returned 403.
- **Game engine updates**: Google News RSS returned no useful Unreal/Unity upscaling update in this run.

## Action Items / Follow-ups
- [ ] Ops: fix/restart trainer or update public status copy; dashboard has been stuck at step 30,300 for 53h while still saying “live/training,” and GPU stats are unavailable.
- [ ] Deep-read **Lift4D** for occlusion-aware persistent Gaussian recovery; decide whether any teacher-side loss/initialization idea transfers to OSS disocclusion handling.
- [ ] Skim **GaussianZoom** for continuous LOD/visibility modulation that could map onto v7 parent-child density or packed-canvas budgets.
- [ ] Validate **MedGSSR** repo completeness; ignore domain, inspect hierarchy/feed-forward GS-SR mechanics if implementation is nontrivial.
- [ ] Watch NVIDIA DLSS 4.5 dynamic multi-frame generation and AMD FSR 4.1 RDNA3 parity claims for competitive messaging.

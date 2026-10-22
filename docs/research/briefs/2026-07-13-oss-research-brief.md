# OSS Research Brief — 2026-07-13

## Project Status Summary
- Latest commits: `a97a180 feat: add tailnet-gated training controls to dashboard`; `fcf8084 feat: tailnet-gated training control API + dashboard panel`; `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`; `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`; `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`; plus flash-attn, TF32, dashboard, rasterizer, and apex redirect fixes in the last 10 commits.
- Open issues: #14 VRAM footprint, #13 Nsight L2 hit-rate profile, #12 packed low-rank canvas layout, #10 G-buffer encoder ablation, #9 RepVGG/ECBSR/RepSR student ablation, #8 cross-vendor kernel ports, #7 FSR 4 binary distillation, #6 v6.2 launch cleanup, #5 no-ML shader variant, #4 HAT-Tiny student distillation.
- README snapshot: still pre-alpha; README headline describes the unified persistent Gaussian canvas / covariance-resampling path and emphasizes dashboard runs are research/teacher models, not the shipping ≤1M-param student. README text still lags the dashboard in places by naming v6/v6.1/v6.2 as active while the dashboard shows v7.0 Pico OSS-FX.
- Training dashboard: rendered page redirects to `opensupersampling.org` and shows `srcnn-v7.0-pico-005` / v7 at step `35,200 / 100,000`, loss `0.19525`, viz at step `26,000`, `35.2%` complete, ETA `~1.1d`, cost `$0.88`, GPU-hours `14.67`, kWh `5.87`. The page headline says “v7.0 Pico OSS-FX is training right now” and badge says `live`, but service health tooltip says **Trainer — offline, stuck at step 35200 for 42h**. Watcher healthy, published `1s` ago; CF Worker healthy, HTTP 200 in `87ms`; R2 origin healthy, HTTP 200. GPU stats unavailable (`host offline or rsync failed`). Compared with 2026-07-12/earlier same-day state, step/loss did not advance and stuck duration increased from 18h → 30h → 42h; treat the live/ETA copy as stale until the trainer heartbeat is fixed.

## New Papers (arXiv)

Note: arXiv Atom API timed out during the broad scheduled pass, so this run used web/search results plus direct arXiv abstract extraction for candidate IDs. Papers below are mostly **newly resurfaced / already-tracked material leads**, not necessarily newly submitted today; no duplicate bibliography append was made for entries already present in `BIBLIOGRAPHY.md`.

### Gaussian Splatting / Temporal
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567, Nengbo Lu, Bin Zhao, 2026-05-11) — Learns per-Gaussian velocity fields with physics codes, a particle dynamics system, and global physical constraints for dynamic multi-view videos. Why it matters to OSS: direct analogue for v7 native time-axis / per-Gaussian motion-state design and future-frame extrapolation losses; already tracked in bibliography.
- **SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting** (arXiv:2602.24020, Xiang Feng et al., 2026-02-27) — Feed-forward sparse-LR-view → HR-3DGS mapping with Gaussian offset learning and feature refinement. Why it matters to OSS: supports the broader thesis that high-frequency Gaussian detail can be learned feed-forward instead of per-scene optimized, but it is 3D multi-view reconstruction rather than 2D game-frame runtime SR; already tracked in bibliography.
- **SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting** (arXiv:2512.02172v2, Pranav Asthana et al., 2025-12-01 / revised 2026-03-28) — Selectively injects SR content only into under-supervised 3DGS regions using pose/geometry cues to avoid multi-view inconsistency. Why it matters to OSS: strong conceptual fit for confidence-/coverage-gated teacher detail injection into a persistent canvas; already tracked in bibliography.

### Super-Resolution / Upscaling
- **Arbitrary-Scale 3D Gaussian Super-Resolution** (arXiv:2508.16467v2, Huimin Zeng, Yue Bai, Yun Fu, 2025-08-22 / revised 2025-11-19; AAAI 2026) — Integrated scale-aware 3DGS rendering plus generative-prior-guided optimization and progressive super-resolving for integer and non-integer scale factors, reporting 85 FPS at 1080p. Why it matters to OSS: useful scale-aware rendering reference if OSS wants non-integer / dynamic resolution targets rather than fixed integer game upscales; already tracked in bibliography.
- **Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution** (arXiv:2605.22147, Jiangwei Mo, Xi Lu, Hanlin Wu, 2026-05-21) — Combines flow matching, shortcut consistency, and 2D Gaussian splatting as a continuous feature field for arbitrary-scale remote-sensing SR. Why it matters to OSS: off-domain, but directly relevant to continuous query-location reconstruction and faster generative detail priors; already tracked in bibliography.
- **ConFi-GS Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution** (arXiv:2605.24964, Jiaxiang Li et al., 2026-05-24) — Confidence/reliability maps gate where high-frequency SR details are injected. Why it matters to OSS: remains a high-priority deep-read for rejecting hallucinated teacher detail before it contaminates the v7 Gaussian canvas; already tracked.

### Frame Extrapolation / Generation
- **ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors** (arXiv:2603.26835v3, Shibo Liu, 2026-03-27 / revised 2026-04-01) — Reuses H.264 decoder motion vectors to prealign frames and removes learned optical flow / spatial sampling / iterative accumulation from the accelerator graph, reporting 12.8 ms 1080p INT8 inference on Snapdragon 8 Gen 3. Why it matters to OSS: very strong deployment analogue for OSS-FX’s “trust externally provided motion vectors, keep the neural residual cheap” path; useful benchmark for motion-vector-prior extrapolation and quantization failure modes; already tracked in bibliography.
- **MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation** remains the current VFI deep-read target from the 2026-07-12 brief for nonlinear motion / occlusion handling; no newer primary VFI paper materially displaced it today.

### Inference Optimization
- Search surfaced no fresh TensorRT / INT8 / kernel-fusion SR primary paper beyond already-tracked items. The current inference risk remains practical: exported FP8/INT8 is not enough unless the ≤1M-param student graph maps to real fused, compute-bound kernels without memory-bound sampling bottlenecks.
- ANVIL’s operator-level claims are relevant here too: iterative flow/refinement can collapse under 8-bit PTQ due to quantized recurrent accumulation, and spatial sampling can be a deployment blocker on constrained accelerators.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad Archive.org hits, all broad GitHub/star-list mirrors (`pbsds-starred`, `awesome-image-registration`, etc.); no new primary paper artifact found.
- `"frame generation" "deep learning"`: 3 hits including `DLSS Swapper builds for Windows` and an OSF registration for `MoCha-LD: Temporal-Chunked Latent Diffusion...`; archive/software leads only, not direct OSS architecture references.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16k+ noisy results dominated by unrelated “grape” media/legal items. Broad query is unusable; continue targeted Wayback checks by exact paper/project URL.
- Targeted Wayback CDX checks were attempted for known arXiv/repo URLs (`2501.13255`, `2407.02677`, `ResonantBraneSplatting`, `RE-of-FSR-4.1.0-Upscaling`); no new actionable archival signal was surfaced in this run.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — Python, pushed 2026-07-01, 0 stars. Still the main fresh GS-SR repo lead; claims arbitrary-scale SR with “Branes” / higher-order primitive behavior. Opportunity: primitive-level ablation for high-frequency detail per splat if raster cost is bounded.
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — Python, pushed 2026-07-02, 4 stars; found under broader `neural upscaler` search, while exact `FSR 4 upscaling reverse engineering` returned 0. Threat/opportunity: if legitimate, could expose FSR 4.1 structure/weight-layout details; verify provenance before citing technically.
- `frame generation open source` returned **OULDRABAH95/AllPen-framework-** (0 stars, pushed 2026-07-08), but it looks unrelated/low-signal for OSS game frame extrapolation.
- `neural upscaler` also returned low-signal landing-page/doc repos (`cloudy-sky-gold/*`). No credible new open-source game upscaler competitor surfaced beyond the already-known FSR reverse-engineering lead.
- `DLSS FSR analysis tool` returned 0 fresh repos in the 14-day pushed window.

## News
- DLSS: Google News RSS surfaced NVIDIA’s primary **DOOM: The Dark Ages | Revelations Launches Today Including DLSS Multi Frame Generation & Path Tracing** plus commentary/tooling headlines. Competitive signal: NVIDIA continues pairing DLSS/MFG with marquee path-traced game launches; OSS positioning should emphasize vendor-neutral one-architecture SR+FX and inspectable internals.
- FSR: RSS surfaced **FSR 4.1 on RDNA 3** analysis (Digital Foundry / TechSpot), OptiScaler/Vega experimentation chatter, and AMD driver/update chatter. The old `gpuopen.com/fidelityfx-super-resolution/` URL still returns 404; GPUOpen news page itself loads. Treat FSR 4.1 details as headline-level until primary SDK/release notes are manually checked.
- Intel/XeSS: Intel XeSS page returned HTTP 403; RSS results were mostly handheld/benchmark chatter and low-quality aggregator posts. No primary technical XeSS update confirmed.
- Apple/Metal: Apple Metal overview loads, but RSS only surfaced generic Mac gaming/performance commentary. No concrete MetalFX/upscaler API change confirmed.
- Unreal/Unity: Unreal blog returned HTTP 403; Unity releases page loads. No confirmed engine-native upscaling integration change found in this run.

## Action Items / Follow-ups
- [ ] Ops: fix/verify trainer heartbeat. Dashboard says live/training with ETA, but trainer health says offline/stuck at step 35,200 for 42h and GPU stats are unavailable.
- [ ] Deep-read **SplatSuRe** and **ConFi-GS** together for confidence-/coverage-gated teacher detail injection; translate into v7 canvas pseudo-label acceptance rules.
- [ ] Deep-read **ANVIL** for motion-vector-prior interpolation, INT8 failure modes, and “remove sampling from accelerator graph” lessons for OSS-FX.
- [ ] Deep-read **VeloGauss** for per-Gaussian velocity/state regularization that could map to v7 native time-axis and future-frame extrapolation losses.
- [ ] Verify **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** provenance and technical depth; if legitimate, write a competitor note on FSR 4.1 neural-upscaler structure.
- [ ] Manually find the current AMD FSR/FidelityFX SDK release URL; old `gpuopen.com/fidelityfx-super-resolution/` returned 404 again.

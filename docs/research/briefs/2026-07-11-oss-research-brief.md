# OSS Research Brief — 2026-07-11

## Project Status Summary
- Latest commits: `a97a180 feat: add tailnet-gated training controls to dashboard`; `fcf8084 feat: tailnet-gated training control API + dashboard panel`; `0660f55 wipe-prep: preserve v7-pico-005 step 5000 snapshot + recovery doc`; `56a9a65 trainer: revert DataLoader to num_workers=0 (WSL2 deadlock)`; `d1255c8 trainer: WindowsPath shim on Linux for cross-OS checkpoint resume`; plus flash-attn/TF32/dashboard/rasterizer fixes in the last 10 commits.
- Open issues: #14 VRAM footprint, #13 Nsight L2 hit-rate profile, #12 packed low-rank canvas layout, #10 G-buffer encoder ablation, #9 RepVGG/ECBSR/RepSR student ablation, #8 cross-vendor kernel ports, #7 FSR 4 binary distillation, #6 v6.2 launch cleanup, #5 no-ML shader variant, #4 HAT-Tiny student distillation.
- README snapshot: still pre-alpha; v7 is the active research/teacher architecture with native `(x,y,t)` Gaussian canvas, 3D Cholesky covariance / `V_xt` motion, parent-child loss-adaptive density, Mip-Splatting filters, and alpha=0.5 OSS-FX supervision. Shipping ≤1M-param student is still not trained.
- Training dashboard: `opensupersampling.com` redirects to `opensupersampling.org`. Visible state says `v7.0 Pico OSS-FX is training right now`, active run `srcnn-v7.0-pico-005`, step `35,200 / 100,000`, loss `0.19525`, viz `@ step 26,000`, 35.2% complete, ETA `~1.1d`, updated just now/live. Service-health tooltip contradicts the headline: `Trainer — offline`, stuck at step `35200` for `6h`; Watcher healthy published `1s` ago; CF Worker healthy HTTP 200 in `92ms`; R2 origin healthy HTTP 200. GPU stats unavailable / host offline or rsync failed. Compared with 2026-07-10, training step/loss did not advance, while the stuck-duration counter appears to have reset/decreased from the earlier same-day 11h observation to 6h without training progress.

## New Papers (arXiv)

### Gaussian Splatting / Temporal
- **On the Design of Mixture-of-Experts for Dynamic Gaussian Splatting** (arXiv:2607.08250, In-Hwan Jin, Hyeongju Mun, Joonsoo Kim, Kugjin Yun, 2026-07-09) — Studies multi-deformation modeling for dynamic 3D Gaussian representations instead of forcing one deformation model across heterogeneous motion. Why it matters to OSS: direct design reference for v7 if a single `V_xt` / native-time canvas underfits mixed static, rigid, articulated, and transient motion; possible router/MoE ablation for dynamic-object canvas updates.
- **Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery** (arXiv:2607.08408, Tianyi Song et al., 2026-07-09) — Online deformable 3DGS pipeline jointly optimizes camera trajectory and deformable scene representation under noisy/missing priors. Why it matters to OSS: off-domain, but the motion-aware online optimization framing is a weak reference for robust canvas updates when engine motion vectors are wrong or missing.
- **NoDrift3R: Raymap-Guided Coupling for Drift-Robust Unposed Feed-Forward 3D Reconstruction** (arXiv:2607.07168, Xiangyu Sun et al., 2026-07-08) — Targets long-sequence pose drift in feed-forward 3DGS reconstruction. Why it matters to OSS: weak/off-domain, but useful as a warning about accumulated pose/state drift in persistent representations; exact engine motion vectors should be a differentiator.
- **GUSH3R: Everyone Everywhere All at Once as Gaussians** (arXiv:2607.05243, Keito Abe et al., 2026-07-06) — Feed-forward dynamic human-scene Gaussian reconstruction. Why it matters to OSS: already tracked from 2026-07-08; still a transient/dynamic-object canvas lead, not a game SR paper.
- **Resonant Brane Splatting for Arbitrary-Scale Super-Resolution** (arXiv:2606.29453v2, Giulio Federico et al., 2026-06-28 / revised 2026-07-01) — Higher-order Gaussian-Hermite/Brane primitives for ASR. Why it matters to OSS: already tracked; remains the strongest primitive-level ablation candidate for better high-frequency detail per splat.

### Super-Resolution / Upscaling
- **AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution** (arXiv:2607.00987, Geunhyuk Youk et al., 2026-07-01) — Diffusion-prior arbitrary-scale VSR with temporal-consistency handling. Why it matters to OSS: already tracked; possible offline teacher/pseudo-target source, too heavy/stochastic for shipping inference.
- **GSPan: A Continuous Gaussian Primitive Representation for Arbitrary-Scale Pansharpening** (arXiv:2606.17722, Fangyi Li et al., 2026-06-16) — Uses 2D Gaussian primitives as continuous residual/detail representation for arbitrary-scale pansharpening. Why it matters to OSS: already tracked; supports the broader 2D-GS-as-image-operator trend, but remote sensing domain limits direct transfer.
- **teasr: training-efficient any-step diffusion transformer for real-world image super-resolution** (arXiv:2606.16188, Xiang Gao et al., 2026-06-15) — Any-step diffusion SR with training-efficiency focus. Why it matters to OSS: already tracked as possible teacher/pseudo-target recipe, not a runtime path.

### Frame Extrapolation / Generation
- **CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations** (arXiv:2605.10586, Nengbo Lu, Minghua Pan, 2026-05-11) — Gaussian-state future prediction / causal dynamic-scene modeling. Why it matters to OSS: already tracked; useful as a longer-horizon OSS-FX regularization reference beyond one-step alpha interpolation.
- **VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos** (arXiv:2605.10567, Nengbo Lu, Bin Zhao, 2026-05-11) — Learns per-Gaussian velocity fields without explicit physical priors. Why it matters to OSS: already tracked; compare explicit velocity-field supervision against v7 `V_xt` cross-covariance motion encoding.
- **Continuous Space-Time Video Super-Resolution with 3D Fourier Fields** (arXiv:2509.26325v2, Becker et al., 2025-09-30) — Continuous spatial+temporal video representation with analytic anti-aliasing. Why it matters to OSS: already tracked; strong non-Gaussian comparison for continuous time-axis rendering.

### Inference Optimization
- **Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training** (arXiv:2604.20291, Pham Phuong Nam Nguyen et al., 2026-04-22) — Quantized SR student with deployment-aware training and teacher guidance. Why it matters to OSS: already tracked; directly relevant to ≤1M-param INT8/FP8 student distillation.
- No newly surfaced TensorRT/kernel-fusion SR paper in this scan beat the already-tracked quantization/kernel leads. Keep the focus on deployment-aware SR quantization, Decoder-Free Distillation, QuantSR+, and fused low-bit kernel references from prior briefs.

## Archive.org Findings
- `"gaussian splatting" "super-resolution"`: 3 broad archive.org hits, all GitHub/archive-list mirrors; no confirmed archived primary paper/code replacement for GS-STVSR/GSASR/GRAPE/STSS.
- `"frame generation" "deep learning"`: 4 broad hits. Notable archived software/build pages include `dlssswapperbuildswin` and `fluidframeswin`; no primary research artifact.
- `"real-time upscaling" game`: 0 hits.
- `GS-STVSR OR GSASR OR GRAPE OR STSS`: 16,617 broad hits, polluted by unrelated `GRAPE` media/items; not useful as evidence.
- Targeted Wayback CDX checks returned confirmed `github.com/ChrisDud0257/GSASR` snapshots (`20250313053211`, `20250801002017`, `20250804114800`). No snapshots returned this run for `arxiv.org/abs/2604.18047`, GS-STVSR GitHub, GRAPE GitHub, or NVLabs/STSS; treat as inconclusive, not proof of absence.

## GitHub / Open Source
- **GiulioFede/ResonantBraneSplatting** — Python, pushed 2026-07-01, 0 stars, no license field. Still the main fresh GS-SR code lead; inspect once implementation stabilizes because Brane primitives may reduce splat count for high-frequency detail.
- **pranav-asthana/SplatSuRe** — Python, 83 stars, pushed 2026-06-16, already in bibliography. Still the most material open GS-SR repo lead after RBS; selective / region-gated expensive operations remain relevant for OSS canvas detail recovery.
- **Azusa309/MedGSSR** — Python/MIT, 2 stars, pushed 2026-06-22. Off-domain medical GS-SR; hierarchical feed-forward Gaussian allocation remains a possible allocation-pattern reference.
- **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** — Python/MIT, 4 stars, pushed 2026-07-02. Description claims static structural analysis of AMD FSR 4.1.0 neural upscaler, DXIL entrypoints, weight extraction, blob layout, strict verification, and rebuild comparison. Threat/opportunity: if real, it is a competitor-analysis shortcut for FSR 4.1 architecture/weights; verify provenance before using.
- **Devanshu-dev/optiscaler-client-fsr4** — C#/MIT, 150 stars, pushed 2026-06-25. More integration/mod-manager signal than core model tech, but shows fast public ecosystem movement around FSR4 swapping.
- **4BH15H3K-SB/optiscaler-fsr4-render-harness**, **Spiccatobigeye413/optiscaler-client-fsr4**, **BigBoiM7MD/pd-perfmod-fsr4** — new/updated 2026-07-10/11 FSR4/OptiScaler/mod harness leads with 0 stars. Likely low-quality until inspected, but public FSR4 tooling churn is accelerating.
- `frame generation open source` GitHub search returned only unrelated/low-signal repos this run; no new credible open-source frame-generation competitor surfaced.

## News
- DLSS: primary Google News lead from NVIDIA says **DOOM: The Dark Ages | Revelations Launches Today Including DLSS Multi Frame Generation & Path Tracing** (2026-07-07). Competitive signal: DLSS 4/MFG continues expanding via high-profile game content drops, so OSS messaging should emphasize vendor-neutral unified SR+FX, not merely “has frame gen.”
- FSR: Digital Foundry headline says **FSR 4.1 on RDNA 3: Useful for Steam Machine, Transformative for More Powerful AMD GPUs** (2026-07-09); TechSpot and CG Channel headlines also point to FSR 4.1 / SDK 2.3 running beyond the newest AMD hardware. Primary GPUOpen deep-read still needed. Competitive signal: AMD compatibility story may widen before OSS ships.
- Intel/XeSS: RSS mostly MSI Claw / Arc G3 handheld benchmark chatter and low-quality mirrors; no primary Intel XeSS technical update confirmed because Intel page fetch hit HTTP 403.
- Apple/Metal: AppleMagazine/MacRumors-style headlines discuss Metal gaming and Apple Silicon demand, but no concrete Metal upscaler/performance API change confirmed.
- Unreal/Unity: no new engine-native upscaling integration change confirmed in primary pages. Unreal feed was not parsed as a useful primary source; Unity keyword scan surfaced no DLSS/FSR/XeSS/upscaling hits.
- Vendor-page checks: NVIDIA Technical Blog DLSS tag and GPUOpen News were reachable; NVIDIA consumer DLSS news page was accessible via Google News lead; Intel XeSS page was HTTP 403 in the collector.

## Action Items / Follow-ups
- [ ] Ops: dashboard contradiction remains the top issue — public copy says live/training with ETA, but trainer health says offline stuck at step 35200 for 6h and GPU stats are unavailable. Since step/loss did not advance from 2026-07-10 and the stuck-duration counter appears reset/inconsistent, verify trainer heartbeat and host state before relying on ETA.
- [ ] Deep-read **On the Design of Mixture-of-Experts for Dynamic Gaussian Splatting** for a possible motion-router / deformation-expert ablation if v7 `V_xt` struggles with heterogeneous dynamic objects.
- [ ] Deep-read / clone **ResonantBraneSplatting** once code is usable; compare Brane primitive expressiveness against v7 splat count, raster cost, and edge/detail recovery.
- [ ] Verify **Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling** provenance and technical depth; if legitimate, write a competitor note on FSR 4.1 neural-upscaler structure and extracted weight/blob layout.
- [ ] Check primary GPUOpen/FidelityFX SDK release notes for FSR 4.1 / SDK 2.3 compatibility details instead of relying on RSS headlines.
- [ ] Replace broad Archive.org acronym searches with exact CDX checks and retry CDX with smaller per-URL calls; current Wayback calls timed out.

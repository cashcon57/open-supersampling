# Report-derived action backlog

**Date:** 2026-06-12  
**Revision:** v3 — refreshed with research reports landed in the last 5 days; still corrected for operator feedback that **v6/v6.1/v6.2 are legacy baselines, not active development targets.**  
**Scope:** synthesized from coordination docs, experiment reports, roadmap/status docs, the v7 pivot recorded in `CHANGELOG.md`, and new research notes through 2026-06-12.

## Correction: why v6 work appeared, and why it should not be active

The first pass over-weighted old reports from 2026-05-08 because there are many detailed v6/v6.1/v6.2 memos. That was stale prioritization.

Current repo history says the project moved on: `srcnn-v6.2-pico-002` was stopped at 74K and **not resumed** because v7 changed the architecture substantially: N-D Gaussian primitive, V_xt cross-correlation in the Cholesky covariance, parent-child loss-adaptive density, OSS-FX time-slice rendering, and Mip-Splatting anti-aliasing filters. The changelog explicitly says marginal v6.2 training would not validate any of that.

So: **do not spend new training/model effort on v6.** Keep v6/v6.2 only as historical baselines, regression references, and source of reusable implementation lessons.

Sources:
- `CHANGELOG.md:29-35` — v6.2 stopped, v7-pico-005 launched, dashboard default switched to v7.
- `CHANGELOG.md:37-45` — v7 Phase 2 closed; v7 model/trainer/tests wired; Phase 3 is v7 training.
- `CHANGELOG.md:20-23`, `Starting up after wipe.md:97-100` — v6 runs are archived as legacy metrics, not active checkpoint lines.

## Last-5-day research delta

Repo/cron scan at 2026-06-12 23:23 CDT found these research inputs landed since 2026-06-08:

- `docs/research/hypotheses/H011-v7-thin-structure-error-residuals.md` — filed 2026-06-08 from manual dashboard viz inspection around `srcnn-v7.0-pico-005` step ~19K. It classifies bright crossbar/mullion patterns in the `|err v7 alpha=0.5|` panel as likely thin-structure edge-alignment / contrast residuals, **not** duplicated geometry in RGB and **not** a training blocker by itself. Escalate only if RGB duplication, flicker/crawl, worsening metrics, or broad recurrence appears. Sources: `docs/research/hypotheses/H011-v7-thin-structure-error-residuals.md:5-12`, `:26-45`.
- `docs/research/briefs/2026-06-08-oss-research-brief.md` — surfaced `Liquid Neural Networks ... Dynamic 3DGS`, `EvoGS`, `SparseStreet`, dynamic-4DGS paradigm survey, `LEGS`, `GS-STVSR`, `3DGS^3`, `VeloGauss`, `R5DGS`, `ReFrame`, deployment-friendly SR transformers, hybrid precision, WebGPU dispatch overhead, OptiScaler, FSR 4/Redstone, DLSS 4.5, and MetalFX/Metal 4 signals.
- `docs/research/briefs/2026-06-09-oss-research-brief.md` — added `REFINE` render-free 3DGS pruning, `Latent Spatial Memory`, `LiteVSR`, `NGram-MoSE`, `Ultra Flash`, and a first strong `RDNU` competitor signal. It also flagged dashboard/trainer stall as the ops blocker.
- `docs/research/briefs/2026-06-10-oss-research-brief.md` — added `Envision4D`, `ManiSplat`, `QuantSR+`, `AutoMegaKernel`, fixed-point/NPU restoration, AI-LOD precision scheduling, compressed-domain online VSR, and repeated RDNU/DLSS/XeSS competitive pressure. Source actions: deep-read RDNU, Envision4D, AutoMegaKernel/QuantSR+, ablate LEGS, investigate AI-LOD scheduling.
- `docs/research/briefs/2026-06-11-oss-research-brief.md` — reinforced `PersistGS`, `R5DGS`, `ConFi-GS`, `Flow-based Gaussian Splatting`, INT8 deployment-aware SR, and RDNU; called out DLSS 4.5 Ray Reconstruction, AMD FSR 4 support-segmentation narrative, and Intel XeSS 3 MFG.
- `docs/research/briefs/2026-06-12-oss-research-brief.md` — consolidated top follow-ups: `VeloGauss`, `Liquid-D3DGS`, `PersistGS`, `3DGS^3`, RDNU, SR3R, NVIDIA DLSS 4.5/Ray Reconstruction, AMD RDNA4 WMMA/tooling, Intel XeSS 3 MFG, INT8/QAT, TensorRT/dispatch-fusion, and targeted Wayback CDX checks for prior-art URLs. Latest live dashboard check from this update showed v7 at step `23,950`, viz tail through `step-00023000.png`; prior stall appears cleared, but ops monitoring remains P0.

Backlog impact:

1. Keep P0 focused on live v7 truthfulness/training because multiple briefs flagged trainer/dashboard stalls as the only blocker that invalidates research progress.
2. Add a v7 research triage bundle: velocity/time-axis priors, object permanence/rigid semantic classes, pruning/active-set, LEGS/detail loss, and thin-structure diagnostics.
3. Add a competitor/baseline audit: RDNU, 3DGS^3, GS-STVSR, OptiScaler, DLSS 4.5/FSR 4/XeSS 3.
4. Expand v7-native performance work with quantization/fusion/dispatch-specific findings: QuantSR+, deployment-aware INT8 SR, AutoMegaKernel, WebGPU dispatch overhead, AI-LOD scheduling, RDNA4 WMMA.

## P0 — v7 active-run correctness and dashboard truthfulness

1. **Fix dashboard `loss --` for v7.**
   - Add `latest_metrics.total` fallback anywhere dashboard headline/hero reads `loss_total ?? loss`.
   - Highest-priority because production showed `step 500 / loss --` even though live data had `latest_metrics.total = 0.3094`.
   - Sources: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:25`, `:32-40`, `:61`, `:99-106`.

2. **Add dashboard data-shape guard for latest metrics.**
   - Add `tools/check_data_schema.py` assertion that every `runs[*].latest_metrics` exposes one of `loss_total`, `loss`, or `total`.
   - Gate dashboard pushes with this plus screenshots.
   - Source: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:93-104`.

3. **Resume/verify the active v7-pico-005 Linux training path.**
   - Active run is `srcnn-v7.0-pico-005`, resumed from preserved step 5000 snapshot.
   - Compare post-resume step 5050+ metrics against `archive/v7-pico-005-snapshot-2026-05-16/history.jsonl` to detect host-transition regressions.
   - Sources: `Starting up after wipe.md:5`, `:12`, `:32`, `:227-265`, `:347-359`; `CHANGELOG.md:10-27`.

4. **Keep dashboard default and public narrative centered on v7.**
   - v6.2 should remain labeled as stopped/baseline-for-v7, not as a thing waiting for more work.
   - Sources: `CHANGELOG.md:29-35`, `:49-53`.

## P1 — v7 implementation / data path

5. **Continue v7 Phase 3 training/evaluation, not v6 retraining.**
   - v7 Phase 2 already closed with end-to-end training-step integration; Phase 3 is the full v7-pico-005 training run.
   - Sources: `CHANGELOG.md:37-45`.

6. **Use archived v6/v6.2 metrics only as baselines.**
   - Keep legacy runs for historical analysis and dashboard comparison.
   - Do not launch v6.1/v6.2 post-fix runs, v6.2-pico-light ablations, or v6 Phase 4 quality tests unless explicitly reframed as a v7-relevant experiment.
   - Sources: `CHANGELOG.md:20-23`, `CHANGELOG.md:29-35`, `Starting up after wipe.md:97-100`.

7. **Prepare v7 data-channel expansion when Cyberpunk captures land.**
   - Next training cycle from Cyberpunk captures should expand to 11 channels: captured stencil as-is plus derived NoV; widen v7 backbone `in_channels=9` to `11`.
   - Source: `docs/capture/v7-training-data-spec.md:67`.

8. **Add v7 dashboard event/annotation support.**
   - Keep the useful lesson from v6 loss-retune: dashboard needs an event layer so retunes, restarts, host migrations, and training anomalies are visible.
   - Implement as v7/general dashboard infra, not v6-specific followup.
   - Legacy source for the lesson: `docs/superpowers/experiments/2026-05-08-v6.1-plateau-loss-imbalance-diagnosis.md:149-159`.

9. **Add a thin-structure diagnostic slice only if H011 escalation criteria trigger.**
   - Current H011 classification is observation/non-blocker: error heatmaps can overstate small edge/contrast offsets around window mullions while RGB output does not show duplicated geometry.
   - If RGB duplication, temporal crawl, worsening metrics, or broad recurrence appears, add an edge-mask / line-structure residual slice over high-gradient GT regions and track it separately from full-frame PSNR/LPIPS.
   - Sources: `docs/research/hypotheses/H011-v7-thin-structure-error-residuals.md:26-45`.

10. **Triage the new v7 Gaussian/time-axis research bundle into concrete ablations.**
   - Velocity/time priors: deep-read `VeloGauss`, `Liquid-D3DGS`, `Envision4D`, `CausalGS`, and `R5DGS` against v7 `V_xt`, exact engine motion vectors, and `t_query` smoothness.
   - Canvas/object persistence: mine `PersistGS`, `RiGS`, `SparseStreet`, `Latent Spatial Memory`, and `EvoGS` for static/rigid/transient primitive classes, occlusion permanence, parent-child density, and active-set pruning.
   - Loss/detail: try LEGS-style Laplacian/detail weighting only as a training/loss ablation, because it does not change runtime architecture.
   - Sources: `docs/research/briefs/2026-06-08-oss-research-brief.md`, `docs/research/briefs/2026-06-09-oss-research-brief.md`, `docs/research/briefs/2026-06-10-oss-research-brief.md`, `docs/research/briefs/2026-06-12-oss-research-brief.md`.

11. **Audit competitor/baseline pressure from last-5-day reports.**
   - `RDNU` is the most relevant new open-source competitor lead: RDNA3-focused, G-buffer-guided neural upscaler, useful as AMD/deployment/baseline pressure even if its claims need verification.
   - `3DGS^3` and `GS-STVSR` remain closest paper collisions for unified SR+FG / 2DGS continuous space-time claims.
   - `OptiScaler` remains the integration/shim precedent, not an architecture baseline.
   - Vendor pressure: DLSS 4.5 Ray Reconstruction, FSR 4/Redstone, XeSS 3 MFG. Position OSS around open/vendor-neutral unified SR+FG, not immediate ray-reconstruction parity.
   - Sources: `docs/research/briefs/2026-06-08-oss-research-brief.md`, `docs/research/briefs/2026-06-10-oss-research-brief.md`, `docs/research/briefs/2026-06-11-oss-research-brief.md`, `docs/research/briefs/2026-06-12-oss-research-brief.md`.

## P2 — capture tool / real training data pipeline

12. **Build the actual D3D12 capture DLL path.**
   - Biggest gap: hook swapchain Present, command queue retention, NGX EvaluateFeature unpack, sampler, GPU→CPU readback, worker EXR writer, JSON config reader, guarded hook callbacks.
   - Needs Windows host + D3D12 game; estimated 2-3 focused weeks.
   - Sources: `docs/coordination/capture-tool-next-phases.md:20-45`.

13. **Finish autonomous capture-tool batches C.1-C.6.**
    - `scripts/capture_simulator.py` operator harness.
    - Fly.io ingest deploy artifacts.
    - Anonymized contributor leaderboard route + dashboard panel.
    - `docs/capture/INSTALL.md` and `docs/capture/SUPPORTED_GAMES.md`.
    - E2E tests for 401/409/413/429/network timeouts, especially 429 backoff/no-delete behavior.
    - Daily R2 `_index.parquet` roll-up.
    - Sources: `docs/coordination/capture-tool-next-phases.md:47-79`, `:89-95`.

14. **Decide capture-tool operator questions.**
    - Fly.io vs Cloudflare Workers for v1 ingest, anti-cheat disclaimer scope, leaderboard privacy, default sampling mode, MSI pipeline location.
    - Sources: `docs/coordination/capture-tool-next-phases.md:81-87`.

## P3 — dashboard UX consolidation

15. **Collapse dashboard visual history into one surface.**
    - Rename global viz to `Visual Progress`; replace per-run viz-strip cards with a button/focus/drawer into the single surface.
    - Stack mobile controls vertically; hide Play when selected run has <2 strips.
    - Sources: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:63-80`.

16. **Move/merge OSS-FX into Visual Progress.**
    - Preferred: make OSS-FX a run/tab inside Visual Progress so there is one evidence player and one control set.
    - Sources: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:81-92`.

17. **Rewrite the hero for mobile-first utility.**
    - Above first mobile screen: one-line v7 status, collapsible "What's new in v7?", latest visual thumbnail.
    - Sources: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:48-62`.

18. **Verify dashboard changes before push.**
    - Required screenshots at 390x844 and 1440x900 plus programmatic data-shape check.
    - Sources: `docs/coordination/2026-05-14-dashboard-layout-audit-verification.md:99-106`.

## P4 — v7/native performance work, only after mapping old CUDA lessons to v7

19. **Audit old v6 CUDA work for reusable pieces, but do not execute v6 Phase 3/4 as-is.**
    - Old reports were for v6.1/v6.2 rasterizer paths. v7 uses a different N-D Gaussian + OSS-FX time-slice architecture, so the right action is a v7-native kernel/perf audit.
    - Reusable candidates may include broad ideas like LUT exp, redundant-computation cleanup, and profiling discipline — but they need v7-specific validation before implementation.
    - Sources for old work: `docs/coordination/cuda-phase4-pipeline-elegance-report.md`; pivot source: `CHANGELOG.md:29-45`.

20. **Create a v7-specific perf baseline/profiler plan.**
    - Do not run old “production-shape v6 end-to-end benchmark” as an active milestone.
    - Instead, benchmark v7 `oss/sr/v7/` forward/training-step surfaces and identify the actual v7 hot path.
    - Sources: `CHANGELOG.md:37-45`; old v6 benchmark status only for contrast: `docs/coordination/cuda-phase4a-baseline.md`.

21. **Fold new quantization/fusion research into the v7 student/runtime plan.**
    - Quantization/student: deep-read `QuantSR+`, deployment-aware INT8 SR, NAWQ-SR, fixed-point/NPU restoration, and ternary/FPGA upscaler signals before freezing the ≤1M-param student.
    - Fusion/dispatch: use `AutoMegaKernel`, WebGPU dispatch-overhead measurements, and deployment-friendly SR transformer papers as evidence for fused kernels / persistent command buffers / minimal launch count.
    - Scheduling: evaluate AI-LOD-style distance/foveation/region-adaptive precision or splat-budget scheduling after the v7 hot path is measured.
    - Sources: `docs/research/briefs/2026-06-10-oss-research-brief.md`, `docs/research/briefs/2026-06-11-oss-research-brief.md`, `docs/research/briefs/2026-06-12-oss-research-brief.md`.

## P5 — cross-vendor ports

22. **Defer AMD HIP / Intel SYCL until v7-native CUDA/perf shape is known.**
    - Old HIP/SYCL plans target v6 Phase 2/3 sum-composite rasterizer kernels.
    - Do not start them verbatim; rewrite port plans around v7 kernels after v7 perf profiling.
    - Sources: `docs/coordination/amd-hip-port-plan.md:1-8`; `docs/coordination/intel-sycl-port-plan.md:1-8`; `CHANGELOG.md:37-45`.

## P6 — heavy/standard publication path

23. **Treat Heavy as sponsor/cloud-capacity work, not local opportunistic training.**
    - A credible Heavy cycle is not one run: main Heavy, ablations, LR/WD/loss sweeps, Standard/Pico distillation, cross-engine fine-tunes.
    - Prefer B200-class capacity if a sponsor can provide it; value is wall-clock compression and fitting Heavy without ugly sharding.
    - Sources: `docs/coordination/2026-05-11-heavy-cloud-training-cost-estimate.md:99-110`, `:78-85`.

24. **Cross-engine validation remains the real shipping gate for v7.**
    - Require TartanAir + UE5 + Unity/Cyberpunk-style held-out validation before claiming shipping quality.
    - Sources: `README.md:341`; `RESEARCH.md:58-60`; `docs/capture/v7-training-data-spec.md:67`.


## Explicitly demoted / removed from active backlog

- v6.1 stippling fix/retrain.
- v6.2-pico-light distilled/from-scratch/F=32 ablations.
- R=16 vs R=8 v6.2 feature-rank defaults.
- v6 foreground second-pass gating.
- v6 Phase 3d parity as a product milestone.
- v6 Phase 4 frametest / checkpoint stats as direct milestones.
- AMD HIP / Intel SYCL plans that assume v6 rasterizer kernels.

These are kept only as archived context unless Cash explicitly asks to resurrect one for comparison or to port a specific lesson into v7.

## Suggested execution order

1. Keep live v7 training/dashboard truthful first: metric fallback, schema guard, viz freshness, and trainer-stall detection.
2. Verify/resume v7-pico-005 from the step-5000 Linux snapshot and compare against archived trajectory.
3. Triage the last-5-day research bundle into bounded v7 ablations: velocity/time-axis priors, object permanence, active-set/pruning, LEGS/detail loss, and H011 diagnostics.
4. Audit RDNU + 3DGS^3 + GS-STVSR + OptiScaler as competitor/baseline pressure before changing public positioning.
5. Finish dashboard UX consolidation around v7/default active run.
6. Finish capture-tool autonomous C.1-C.6 and plan the D3D12 hook for real Cyberpunk/v7 data.
7. Create a v7-native perf/profiling plan; fold QuantSR+/INT8/AutoMegaKernel/WebGPU-dispatch/AI-LOD findings into student/runtime design.
8. Rewrite cross-vendor port plans around v7 after the v7 hot path is known.

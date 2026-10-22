# OpenSuperSampling — Reconciled Research-to-Implementation Backlog

Source corpus:
- OSS research reports in `docs/research/2026-05-28-oss-research-brief.md` through `docs/research/2026-06-08-oss-research-brief.md`.
- Exported Jun 3–8 cron/report history from pasted `message.txt`.
- Two follow-up synthesis reports from pasted `message.txt` / `message2.txt`.

Purpose: reconcile all reports into one implementation backlog, ranked by applicability and expected benefit.

---

## Reconciled thesis

The latest report history changes the priority order. Three things are now unambiguous:

1. **Trainer recovery is the urgent operational blocker.** The active `srcnn-v7.0-pico-005` run repeatedly stalls while the publish path looks healthy. No training or quality conclusion is valid until trainer liveness, GPU telemetry, and metric freshness are instrumented.
2. **GS-STVSR / GSASR / ContinuousSR are the direct technical collision set.** These are not generic prior art; they are the baseline cluster OSS must reproduce, absorb, or explicitly beat.
3. **OSS positioning must shift from “vendor-neutral SR” to “open/moddable unified SR+FG Gaussian canvas with confidence-gated persistent state.”** FSR 4/4.1 and XeSS 3 reduce the strength of “vendor-neutral” alone; openness, moddability, persistent Gaussian state, and unified SR+FG are the sharper differentiation.

---

## P0 — Do first / blocks everything else

### P0.1 Recover stalled `srcnn-v7.0-pico-005` trainer

Repeated evidence:
- step 9,450 stuck for ~3h;
- step 10,300 stuck for ~7h;
- brief recovery at 11,550 / 12,650 but degraded/stuck;
- step 17,700 stuck/offline for 6h → 19h → 30h → 35h;
- watcher / CF Worker / R2 healthy;
- GPU telemetry unavailable or missing.

Likely failure points:
- 3080 Ti trainer host;
- trainer process;
- GPU/CUDA path;
- rsync / metric publishing;
- telemetry gap.

Implement:
- trainer-host health probe;
- trainer process/PID heartbeat;
- stale-step detector: global step/loss freshness alert after >30 min;
- GPU telemetry probe: utilization, memory, driver visibility, CUDA errors;
- rsync / metrics-publisher freshness probe;
- auto-resume from latest checkpoint where safe;
- Discord/ntfy alert with exact failed subsystem;
- dashboard failure states:
  - `trainer_process_dead`
  - `gpu_host_unreachable`
  - `gpu_telemetry_missing`
  - `trainer_stale_step`
  - `rsync_stale`
  - `metrics_publisher_stale`
  - `cloudflare_r2_issue`
  - `infra_ok_but_trainer_stale`

Acceptance:
- dashboard cannot show “healthy” when step/loss is stale;
- alert identifies subsystem, not just “dashboard degraded”;
- latest checkpoint + last metric timestamp are visible;
- restart/resume path is documented or automated.

Why: no architecture conclusion is trustworthy while the active run is stale.

### P0.2 Fix dashboard / README / metrics status drift

Repeated evidence:
- README and recent commits say v7 / `srcnn-v7.0-pico-005`.
- Public dashboard sometimes reports stale v6.1 Pico at 0 / 100,000 waiting for `metrics.json`.
- `score_log_v7.json` absent in at least one report.
- Static dashboard has Tailscale-gated training controls, but freshness/status conflict.

Implement:
- single source of truth JSON for active run:
  - model/run name;
  - architecture version;
  - checkpoint path/hash;
  - step;
  - loss;
  - trainer host;
  - status;
  - last metrics timestamp;
  - last held-out score timestamp;
  - dashboard publish timestamp;
- README generated from status JSON or manually updated with explicit timestamp;
- dashboard displays source + freshness for each field;
- CI/check script flags stale docs vs active run state;
- hard warning if `score_log_v7.json` is missing for active v7 runs.

Why: reports keep wasting cycles reconciling conflicting active-architecture claims.

### P0.3 Read/reproduce GS-STVSR immediately

Repeated evidence:
- GS-STVSR (`arXiv:2604.18047`) is repeatedly called the closest direct prior/collision.
- Direct overlap: 2D Gaussian + continuous space-time video SR.
- It may already solve a large chunk of OSS v7’s intended technical core.

Implement:
- deep-read notes;
- minimal architecture diagram;
- component diff against OSS v7:
  - 2D Gaussian representation;
  - temporal/covariance modeling;
  - optical-flow-guided Gaussian motion;
  - covariance-resampling alignment;
  - continuous space-time rendering;
  - training losses;
  - runtime assumptions;
- decision memo: copy/adapt, benchmark against, or differentiate away from;
- baseline harness adapter if runnable.

Acceptance:
- table mapping GS-STVSR modules to OSS v7 modules;
- explicit novelty/difference list;
- benchmark plan or reproduction blocker list.

Why: if GS-STVSR already solves 70% of v7, OSS should absorb it or beat it explicitly.

### P0.4 Validate GSASR / GaussianSR / ContinuousSR baseline cluster

Direct GS-SR collision set:

| Baseline | What to extract |
|---|---|
| **GSASR** (`arXiv:2501.06838`) | CUDA scale-aware rasterizer, arbitrary-scale 2DGS SR, density/raster path |
| **ContinuousSR** | “build Gaussian field once, render arbitrary scale ~1ms” latency claim, Pixel→Gaussian field construction |
| **GaussianSR / SRGS** | Gaussian SR representation + density strategy |
| **SplatSuRe** | selective SR / updated repo/code / weak-evidence compute allocation |
| **SRSplat** | texture-aware density / adaptive splat strategy |
| **GRAPE** | student / Gaussian SR design ideas |
| **SR3R** | additional GS-SR reference, triage only if implementation-relevant |

Implement:
- benchmark adapters where code is available;
- consumer-GPU latency/VRAM measurement, not datacenter-only claims;
- rasterizer notes for GSASR vs OSS v7 current rasterizer;
- novelty/speed/quality matrix.

Acceptance:
- matrix shows which baselines are runnable, blocked, or paper-only;
- ContinuousSR 1ms claim tested or blocked with reason;
- GSASR rasterizer notes produce concrete kernel/TODO items.

Why: this is the direct “are we novel / are we fast enough” cluster.

### P0.5 Make the canvas confidence-gated

Primary signal:
- ConFi-GS (`arXiv:2605.24964`) — reliability-gated high-frequency injection.
- SuperGS — uncertainty-aware pseudo-label weighting + error-map-guided densification.
- SplatSuRe — selective SR in low-evidence regions.
- MaCo-GAN — LR-faithful contrastive GAN objective.

Problem:
- OSS persistent canvas makes hallucination more dangerous than normal SR. Bad teacher/perceptual detail can become persistent state.

Implement:
- detail-demand map;
- reliability/confidence map;
- pseudo-label filter for teacher outputs;
- splat spawn/update multiplier based on confidence;
- densification gate from uncertainty/error maps;
- LR-faithfulness loss before perceptual-stage fine-tune;
- confidence debug overlay;
- explicit rule: **do not persist unreliable SR detail**.

Correct flow:

```text
teacher/perceptual detail → confidence gate → spawn/update canvas
```

Not:

```text
teacher/perceptual detail → blindly persistent canvas
```

Acceptance:
- confidence map visible in debug tooling;
- ablation: ungated vs confidence-gated teacher detail;
- temporal shimmer/hallucination metrics included.

Why: this is the strongest direct architecture improvement surfaced by the latest reports.

### P0.6 Runtime student deployment: INT8/QAT first-class

Repeated signal:
- Efficient INT8 Single-Image SR via Deployment-Aware Quantization and Teacher-Guided Training (`arXiv:2604.20291`).
- INT8-friendly residual designs from ANVIL-style temporal synthesis.
- TensorRT/CUTLASS/export constraints recur across reports.

Implement:
- PTQ baseline;
- QAT training option;
- teacher-guided INT8 student loss;
- quantize/train the fused deploy graph, not only training graph;
- BN/stat recalibration;
- per-layer regression check;
- INT8/FP8 deployment constraints in student block search;
- hybrid precision exceptions around covariance/composite/normalization if needed;
- target ≤1M-param student.

Acceptance:
- INT8 regression <0.2 dB PSNR vs fp16 target, or documented exception;
- latency measured vs fp16 on target GPU;
- artifact inspection included, not metrics only.

Why: <2ms runtime depends on deployment-aware training, not post-hoc export hope.

---

## P1 — High-benefit architecture / runtime work

### P1.1 Motion/time-axis split for v7 / OSS-FX

Repeated papers:
- RetimeGS;
- RiGS;
- VeloGauss;
- CausalGS;
- FlowGS;
- GeRM;
- LagrangianSplats;
- R5DGS;
- Hybrid 3D-4DGS;
- MotionScale.

Implement architecture experiment:
- persistent splats:
  - stable surfaces;
  - texture;
  - geometry-like residual cache;
- transient splats:
  - disocclusion;
  - speculars;
  - particles;
  - motion blur;
  - unstable detail;
- velocity/time state:
  - splat position/covariance evolves over `alpha`;
  - rigid object / centroid-level motion where possible;
  - learned velocity field vs engine-MV-supervised motion;
- continuous-time interpolation/extrapolation path for OSS-FX;
- physics/causality/divergence-free regularizers only if they improve measured temporal stability.

Acceptance:
- ablation: single splat pool vs persistent/transient split;
- α-conditioned extrapolation metric;
- disocclusion/artifact classification.

### P1.2 Gaussian budget / density control

Signals:
- ZipSplat — fewer Gaussians, better splats;
- SplatSuRe — selective SR;
- SuperGS — uncertainty/densification;
- ConFi-GS — confidence gate;
- SRSplat TADC — texture-aware density control;
- GaussianHSI — Voronoi-guided Gaussian selection;
- packed canvas layout issue cluster.

Implement:
- splat budget controller;
- per-tile/per-region Gaussian budget;
- selective splat spawning;
- confidence-gated densification;
- persistent vs transient density policy;
- packed low-rank canvas layout compatibility;
- profile cache/L2/overdraw with Nsight.

Acceptance:
- budget/quality/latency Pareto curve;
- debug view showing spawn reasons and confidence/density state.

### P1.3 GSASR/ContinuousSR rasterizer wins

Signals:
- GSASR CUDA scale-aware rasterizer;
- ContinuousSR raster path;
- AA-Splat OBBL;
- current v7 Mip-Splatting/fp32 smoothing filters;
- follow-up synthesis calls out tight AABB + conic row recurrence.

Implement:
- inspect GSASR kernels;
- evaluate tight AABB bounds for splat raster;
- conic row recurrence for cheaper per-pixel eval;
- band-limiting + opacity balancing ablation;
- scale-aware raster path;
- compare with current v7 filters.

Acceptance:
- microbenchmarks for raster kernels;
- image-quality regression tests for aliasing/overlap;
- Nsight profile before/after.

### P1.4 CDA-VSR / ANVIL motion-vector-prior temporal synthesis

Signals:
- CDA-VSR — motion-vector-guided deformable alignment;
- ANVIL — codec/exact motion-vector priors + accelerator-native residual design;
- compressed-domain-aware online VSR;
- FAST compute reuse;
- efficient neural supersampling gaming dataset with motion/depth/jitter/mip-bias.

Implement:
- motion-vector-prior alignment branch;
- MV quality validator;
- game G-buffer motion vectors vs learned optical flow comparison;
- codec-style MV fallback where available;
- quantization-friendly temporal residual path;
- temporal residual reuse/skip path when motion/residual evidence is low.

Acceptance:
- ablation: optical flow vs engine MV vs codec MV;
- INT8-friendly temporal residual benchmark;
- quality loss on bad-MV scenes measured.

### P1.5 ReFrame-style temporal feature caching

Signal:
- ReFrame (`arXiv:2506.13814`) — temporal feature caching for real-time rendering workloads.

Implement:
- cache student intermediate features across frames;
- invalidate by motion/disocclusion/confidence;
- compare modes:
  - no cache;
  - full cache;
  - tile-gated cache;
  - motion-gated cache;
- cache hit-rate telemetry;
- p50/p99 latency and temporal error metrics.

Why: could reduce recurrent student cost without changing visual model.

### P1.6 Wavelet/frequency-domain supervision

Signals:
- Wavelet-space rendering SR;
- WaveletSR;
- SlimDiffSR frequency-separable convs.

Implement:
- wavelet-space loss;
- optional wavelet output head;
- frequency-band guided densification/HF injection;
- compare temporal shimmer vs sharper detail.

Acceptance:
- ablation against confidence-gated HF injection;
- frequency-domain metrics plus visual artifact review.

### P1.7 Distillation/loss mining pass

Actionable candidates:
- DUO-VSR;
- DLoRAL-VSR;
- OSDEnhancer;
- MaCo-GAN;
- StreamSR/EfRLFN;
- AceVFI;
- DiffST;
- STSS;
- SlimDiffSR;
- HAT-Tiny;
- FSR4 teacher.

Extract only if it changes training code:
- temporal consistency loss;
- distillation loss;
- low-rank adaptation / ConvLoRA;
- degradation-aware loss;
- LR-faithful contrastive loss;
- frequency-domain supervision;
- efficient student block;
- export/runtime note.

Avoid: broad paper pile with no concrete primitive/loss/runtime trick.

### P1.8 OSS-FX frame pacing / generated-frame multiplier policy

Vendor pressure:
- DLSS 4.5 Dynamic MFG 5X/6X;
- XeSS 3 MFG;
- FSR 4/Redstone;
- GFFE as lower-integration FG foil.

Implement OSS-FX eval dimensions:
- source FPS → target FPS;
- generated-frame multiplier:
  - 1 generated;
  - 2 generated;
  - 3 generated;
  - 5 generated / DLSS 4.5-style pressure;
- α schedule:
  - `α=0.25`;
  - `α=0.5`;
  - `α=0.75`;
  - `α=0.9`;
- latency budget:
  - render frame latency;
  - generated frame latency;
  - queueing latency;
  - input-to-photon penalty;
- artifact classes:
  - disocclusion;
  - HUD/text;
  - particles;
  - camera cuts;
  - fast pan;
  - thin geometry.

Acceptance:
- policy doc says when OSS-FX chooses interpolation, extrapolation, lower multiplier, or disables generation;
- benchmark includes generated-frame quality and frame pacing, not just reconstruction metrics.

---

## P1 — Product/runtime strategy

### P1.9 OptiScaler-style shim / mod deployment path

Repeated signal:
- OptiScaler appears in almost every report as threat and integration reference.
- Related tools/repos: `dlssg-to-fsr3`, `dlss-compositor`, LSFG-Android, NodeNestor/prism-optiscaler.

Implement:
- map OptiScaler interception model;
- inspect config flow and game compatibility conventions;
- identify input contracts:
  - DLSS SR path;
  - DLSS FG path;
  - FSR2/FSR3/FSR4 path;
  - XeSS path if feasible;
  - NGX / Streamline-style interfaces;
- build OSS shim spec:
  - `nvngx_dlss.dll`;
  - `nvngx_dlssd.dll`;
  - Streamline-style adapter;
  - FSR path;
  - XeSS path if feasible;
- add DLSS/NGX/Streamline-style input-contract validator.

Why: OSS distribution lives or dies on drop-in installability.

### P1.10 Treat GFFE as competitor/foil

Signal:
- GFFE (`arXiv:2406.18551`) — low-latency G-buffer-free frame extrapolation.

Implement:
- compare OSS-FX with and without G-buffer inputs;
- measure integration friction:
  - color-only path;
  - depth+motion path;
  - full G-buffer path;
- benchmark whether OSS’s extra integration cost buys enough quality.

Positioning question:
- If GFFE is “good enough” with lower integration friction, OSS-FX must justify hook/G-buffer complexity.

### P1.11 UE5 / G-buffer capture expansion

Signal:
- OSS needs better engine-signal capture to justify exact-MV/G-buffer approach.

Implement:
- UE5 capture path notes;
- required buffers matrix:
  - color;
  - depth;
  - motion vectors;
  - jitter;
  - exposure;
  - reactive/transparency masks if available;
  - material/albedo/roughness where possible;
- dataset export format for OSS eval.

Why: makes MV-prior and G-buffer-vs-GFFE comparisons real instead of theoretical.

---

## P2 — Competitive pressure / platform strategy

### P2.1 Benchmark against 2026 vendor baseline, not DLSS 3 / FSR 3

New bar:
- NVIDIA DLSS 4.5:
  - second-gen transformer SR;
  - linear-space model;
  - Dynamic Multi Frame Generation 5X/6X;
- NVIDIA DLSS 5 / unified neural rendering claims;
- AMD FSR 4 / FSR 4.1 / Redstone:
  - GPUOpen docs;
  - UE plugin;
  - possible RDNA3/RDNA2 expansion;
- Intel XeSS 3:
  - SR + FG + low latency + MFG;
- Apple Metal 4 / MetalFX:
  - frame interpolation;
  - path tracing support;
- Arm NSS:
  - mobile neural upscaling and Vulkan ML signal.

Implement benchmark table:
- OSS-SR vs DLSS 4.5 / FSR4 / XeSS 3;
- OSS-FX vs DLSS MFG / FSR FG / XeSS MFG;
- OSS runtime vs MetalFX where Apple target matters;
- open/proprietary, hardware support, input requirements, latency, mod/shim support.

Positioning shift:
- “Vendor-neutral” alone weakens if FSR4.1 expands backward/cross-vendor.
- OSS needs:
  - openness;
  - moddability;
  - per-game adaptation;
  - unified SR+FG canvas;
  - transparent cross-vendor implementation;
  - mod/game-specific LoRA/adapters;
  - confidence-gated persistent Gaussian state.

### P2.2 Apple / Metal backend remains strategic, not urgent

Repeated but not P0:
- MetalFX;
- Metal 4;
- Apple backend;
- neural shader/runtime direction.

Implement later:
- MetalFX input-contract comparison;
- Metal backend feasibility notes;
- Apple-specific neural shader path;
- avoid CUDA/TensorRT-only architecture decisions that block Metal.

### P2.3 Cross-vendor runtime path

Signals:
- DirectX Compute Graph Compiler;
- Vulkan ML / Arm NSS;
- Metal 4;
- cross-vendor kernel issue cluster;
- WebGPU dispatch overhead.

Implement:
- CUDA/TensorRT path for initial speed;
- DirectML/DirectX Compute Graph feasibility;
- Vulkan ML path notes;
- Metal path notes;
- dispatch-count budget and fused-op constraints shared across backends.

---

## P2 — Repo hygiene / planning

### P2.4 Convert issue clusters into milestones

Repeated issue themes:
- VRAM measurement;
- Nsight/L2 profiling;
- packed canvas layout;
- student ablations;
- cross-vendor kernels;
- HAT-Tiny distillation;
- FSR-teacher distillation.

Milestones:
- `v7-instrumentation`: VRAM, Nsight, L2/cache, timing.
- `v7-canvas-runtime`: packed layout, density budget, rasterizer kernels.
- `student-distillation`: HAT-Tiny, FSR teacher, QAT, low-rank/ConvLoRA, MaCo-GAN/STSS.
- `oss-fx-eval`: frame pacing, MFG policy, GFFE foil, artifact taxonomy.
- `shim-runtime`: OptiScaler-style integration, NGX/Streamline contracts.
- `cross-vendor-runtime`: CUDA/TensorRT, DirectML/DirectX, Vulkan, Metal.

### P2.5 Watchlist / low-immediate-applicability

Keep, but do not block core work:
1. **FramePack / diffusion future-frame systems** — teacher/training ideas, not runtime.
2. **FlowGS / diffusion or flow-matched residual priors** — heavyweight teacher augmentation only unless benchmark says otherwise.
3. **GS-NFS / Gaussian streaming/compression** — relevant only if OSS persists/transmits Gaussian state.
4. **Project-FreeGen and other open FG projects** — monitor as competitive threats and benchmark sources.
5. **Arm NSS open models** — use as mobile baseline/distillation reference if released.
6. **Archive.org paper recovery** — repeated reports found mostly noisy/no useful hits; deprioritize unless a specific missing artifact matters.
7. **MambaLiteSR** — only keep for TensorRT/export notes unless a concrete student block emerges.
8. **Low-signal broad arXiv triage dumps** — exclude unless tied to concrete primitive/loss/runtime trick.

---

## Reconciled top-20 execution order

1. **Trainer watchdog/resume + dashboard failure classification.**
2. **Dashboard/README/metrics/status unification**, including `score_log_v7.json` freshness.
3. **GS-STVSR deep-read + component diff against OSS v7.**
4. **GSASR / ContinuousSR / GaussianSR / SplatSuRe baseline reproduction.**
5. **Canvas confidence-gated update path using ConFi-GS ideas.**
6. **Fix canvas load-bearing/spawner behavior** so unreliable teacher detail cannot become persistent state.
7. **CUDA rasterizer wins:** GSASR mining, tight AABB, conic row recurrence, scale-aware path.
8. **INT8/QAT ≤1M-param student path** with deployment-aware fused graph.
9. **STSS + MaCo-GAN + FSR4/HAT teacher loss integration** after stable baseline convergence.
10. **Motion/time-axis split:** persistent/transient/velocity splats.
11. **MV-prior temporal synthesis:** CDA-VSR / ANVIL / compressed-domain ideas.
12. **OSS-FX frame pacing / MFG eval policy.**
13. **Wavelet/frequency loss/head ablation.**
14. **ReFrame temporal feature caching ablation.**
15. **OptiScaler-style shim spec** and NGX/Streamline input-contract validator.
16. **GFFE competitor/foil benchmark** for integration-friction tradeoff.
17. **Vendor benchmark matrix:** DLSS 4.5 / FSR4 / XeSS 3 / MetalFX.
18. **UE5/G-buffer capture expansion** for exact-MV/G-buffer path validation.
19. **VSR/loss paper mining** only for code-changing ideas.
20. **Cross-vendor backend notes:** DirectML/DirectX Compute Graph, Vulkan ML, Metal.

---

## What changed vs the previous backlog

- Promoted **GS-STVSR reproduction/component diff** to P0 instead of generic temporal-retiming work.
- Promoted **GSASR / ContinuousSR / GaussianSR baseline reproduction** to P0 because it is the direct novelty/speed collision set.
- Promoted **confidence-gated canvas persistence** from “good idea” to core v7 safety rule.
- Made **INT8/QAT** P0 because student deployment constraints should shape architecture, not follow it.
- Added missing concrete execution details from follow-up reports:
  - auto-resume from latest checkpoint;
  - Discord/ntfy subsystem-specific alert;
  - DLSS/NGX/Streamline input-contract validator;
  - tight AABB + conic row recurrence rasterizer tasks;
  - STSS + MaCo-GAN + FSR4/HAT teacher loss integration;
  - UE5/G-buffer capture expansion;
  - mod/game-specific LoRA/adapters as differentiation.
- Reframed competitive positioning: not just vendor-neutral, but **open/moddable unified SR+FG Gaussian canvas with confidence-gated persistent state**.

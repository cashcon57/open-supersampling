# OSS-Gaussian — Sprint 4 Plan

**Spec:** `docs/superpowers/specs/2026-05-01-gaussian-temporal-canvas-design.md`
**Master plan:** `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md`
**Architecture doc:** `docs/superpowers/gaussian-network-architecture.md`
**Branch:** `v0.2-dev`
**Estimate:** ~4 weeks (training time dominates ~1.5–2 weeks GPU)
**Cloud cost target:** $50–100 on Lambda (H100 SXM, 80 GB) — see T4.7 for breakdown.

Sprint 4 produces the **Gaussian Param Network** — the lightweight CNN that
predicts (Δposition, Covariance Prior Bank weights, color) per complex tile from
LR frame + G-buffers + canvas state. It depends on Sprint 1 (renderer) and
Sprint 3 (tile classifier), and feeds Sprint 5 (persistent canvas).

---

## Inputs from prior sprints

- **Sprint 1:** `oss/gaussian/renderer/` — differentiable `Rasterizer` +
  `GaussianBatch` (already wired; CUDA preferred, PyTorch reference fallback).
- **Sprint 2 (in flight):** Cyberpunk RenderDoc / live-capture frame dumps used
  as **input distribution only** (no target labels — license).
- **Sprint 3:** complex/simple tile mask. Sprint 4 trains the network on
  complex tiles only (simple tiles bypass the network at inference).

## Outputs Sprint 5 consumes

- Trained `GaussianParamNetwork` checkpoint (`.pt`).
- TensorRT INT8 engine (`gaussian_param_net.engine`) for live inference.
- `OutputHead` decoding the raw tensor → `GaussianBatch` ready for the
  Rasterizer (also reused by Sprint 5's spawn-on-disocclusion path).
- Model card (`docs/superpowers/gaussian-network-model-card.md`).

---

## Key design decisions (from spec + master plan)

1. **Covariance Prior Bank** — 16 fixed entries (default; learnable + size
   ablated in T4.9). Bank vocabulary enumerated in
   `oss/gaussian/network/prior_bank.py` and the architecture doc.
2. **Network architecture** — 4-level U-Net at LR resolution. Encoder widths
   `(16, 24, 32, 48)` for the standard tier. Tier-scaled widths in
   `TIER_CONFIGS`. Tile-aware output head pools the LR feature map down to
   tile resolution and emits `K * (5 + bank_size + 3)` channels per tile.
3. **Per-Gaussian raw channels (22 at standard config)** —
   `Δμx, Δμy, log_scale, rot_offset, bank_logits[16], color[3]`.
4. **Differentiable end-to-end** — network → `OutputHead.decode` → Rasterizer
   → image-space loss. Gradients flow through the renderer's autograd
   (proven in Sprint 1 / T1.5).
5. **Loss** — composite: HDR-aware L1 + 0.1×(1−SSIM) + 0.05×LPIPS +
   0.1×temporal-consistency (warp-based) + 0.001×covariance regulariser
   (entropy floor on bank softmax to prevent always-pick-one collapse).
6. **TensorRT INT8 export** — at end of sprint. Calibration on a 256-frame
   held-out set sampled across Sintel + TartanAir + Cyberpunk.

---

## Files

```
oss/gaussian/network/
  prior_bank.py            T4.1 — CovariancePriorBank (already scaffolded)
  param_net.py             T4.2 — GaussianParamNetwork
  output_head.py           T4.2 — raw tensor → GaussianBatch
  __init__.py              T4.2 — public API
oss/gaussian/train/
  __init__.py
  data.py                  T4.3 — dataloader (Sintel + TartanAir + Cyberpunk + SRGD)
  losses.py                T4.4 — composite loss
  train.py                 T4.5 — training script (single-GPU + DDP)
  ablate.py                T4.9, T4.10 — ablation runner
oss/gaussian/export/
  trt_export.py            T4.12 — TensorRT INT8 export
  trt_calibrator.py        T4.12 — INT8 calibrator
  bench_trt.py             T4.12 — TRT engine perf bench
docs/superpowers/
  gaussian-network-architecture.md     T4.2 — architecture + tier table
  gaussian-network-model-card.md       T4.13 — model card
  gaussian-network-training-report.md  T4.13 — training run summary
tests/gaussian/
  test_network.py          T4.2 — unit tests (already landed)
  test_loss.py             T4.4
  test_data.py             T4.3
  test_trt_parity.py       T4.12 — FP32 vs TRT INT8 numeric parity
configs/gaussian/
  standard.yaml            T4.5 — standard-tier training config
  pico.yaml, lite.yaml,
    ultra.yaml             T4.10 — tier ablations
```

---

## Tasks

> **Status update 2026-05-02:** T4.1, T4.2, T4.3 (data adapters + engine-aliased LR), T4.5 (training script wiring) are landed. T4.6 / T4.11 (Lambda H100 training) are **postponed indefinitely** — cloud spend is out of budget and v0 MVP must come from the 3080 Ti alone. Anisotropic G-buffer-conditioned covariance bias (Decision 2 from the validation memo) was pulled into T4.2 / `OutputHead`. Live training findings are tracked in `docs/superpowers/experiments/2026-05-02-sprint4-smoke-findings.md`.

### T4.1 — Covariance Prior Bank (DONE during planning)

**Status:** Architecture skeleton already landed.
**Files:** `oss/gaussian/network/prior_bank.py`, `tests/gaussian/test_network.py`.

**Goal:** 16-entry default vocabulary with frozen-buffer + learnable variants.
PSD covariance from (sx, sy, θ); softmax weights → per-Gaussian (sx, sy, θ, Σ).

**Verify:** `pytest tests/gaussian/test_network.py -v -k "bank or default_bank"`

**Acceptance:**
- 16 PSD covariance matrices on default construction.
- One-hot weight recovers the entry exactly.
- Learnable mode exposes 3·K trainable parameters; frozen mode has 0.

**Time:** done (~1 day equivalent).

---

### T4.2 — GaussianParamNetwork + OutputHead (DONE during planning, extended 2026-05-02)

**Status:** Architecture skeleton + tests landed. Anisotropic G-buffer-conditioned covariance bias added per validation-memo Decision 2 (commit `da916d0`).
**Files:** `oss/gaussian/network/param_net.py`, `output_head.py`, `__init__.py`,
`tests/gaussian/test_network.py`.

**Goal:** 4-level U-Net at LR. Tile-aware output head emits raw tile-wise
parameter tensor. `OutputHead.decode` produces a `GaussianBatch` ready for
the Rasterizer.

**Goal (2026-05-02 addition):** `OutputHead.enable_gbuffer_bias=True` adds a per-tile (mean normal, mean depth gradient) → 5-feature linear projection that biases the bank softmax toward edge-aligned anisotropic entries. Zero-init projection means enabling the flag is graceful (matches the disabled output until trained). Shared across the K Gaussians per tile.

**Verify:** `pytest tests/gaussian/test_network.py -v`

**Acceptance:** all 25 + 7 (gbuffer-bias) unit tests pass on CPU. End-to-end differentiability test produces non-zero gradients on stem, head, AND learnable bank. CUDA backward verified non-zero on 3080 Ti via `scripts/probe_cuda_grad_flow.py`.

**Time:** done (~2 days base + ~1 day gbuffer-bias extension).

---

### T4.3 — Dataloader (Sintel + TartanAir + Cyberpunk + SRGD)

**Goal:** A single `GaussianTrainDataset` that yields tuples
`(lr_color, depth, motion, normals, canvas_state, hr_target,
 prev_lr, prev_motion)` for training. Reuses existing `oss/data/sintel_fx.py`
patterns. Cyberpunk frames provide INPUT distribution only (no HR target).

**Files:** `oss/gaussian/train/data.py`, `tests/gaussian/test_data.py`

**Steps:**
1. Adapter base class in `data.py` with `__getitem__` returning the tuple.
2. Sintel adapter: extends `oss/data/sintel_fx.py` to also yield depth (Sintel
   provides ground-truth depth) and synthesise normals via depth-gradient.
3. TartanAir adapter: TartanAir has all G-buffers natively. ~25 GB after
   sub-sampling 1 in 4 frames per scene to fit Lambda H100 SSD budget.
4. Cyberpunk adapter: reads frame dumps from Sprint 2. NO HR target — these
   frames contribute only to a domain-randomised input pool used during
   self-supervised pretraining (predict-render LR ↔ self loss).
5. SRGD adapter: synthetic SR pairs. Used purely for the SR signal.
6. Multi-source weighted sampler: mixes the four sources per the table:
   - Sintel: 30% (validation-friendly synthetic)
   - TartanAir: 50% (volume + G-buffer richness)
   - Cyberpunk: 15% (input distribution alignment)
   - SRGD: 5% (SR-only anchor)
7. canvas_state input: at training time, sample either zeros (cold start) or
   the bilinearly-warped previous-frame target (curriculum: start with zeros
   for the first epoch, mix in warped history afterwards).

**Verify:**
```bash
source venv-py312/bin/activate
pytest tests/gaussian/test_data.py -v
python -m oss.gaussian.train.data --inspect 4   # prints first 4 sample shapes
```

**Acceptance:**
- `__getitem__` returns the documented tuple with float32 tensors on CPU.
- Sintel-only smoke run on 16 samples completes in <30 s.
- Cyberpunk samples are filterable via a flag (so the trainer can switch
  off non-supervised data when tracking PSNR-vs-Sintel for graduation).
- Sampler weights are log-printed at startup for the train run.

**Risks:**
- Cyberpunk frames may be too domain-specific; if early validation shows
  PSNR regression on Sintel, drop their weight to 5%.
- TartanAir motion vectors are scene-flow not screen-space — convert via
  the depth + intrinsics once per scene at preprocess time.

**Time:** ~3 days.

---

### T4.4 — Loss design + ablation

**Goal:** Composite loss. Each term tested in isolation before combination.

**Composite:**
```
L_total = L_hdr_l1
        + 0.1  * L_ssim
        + 0.05 * L_lpips
        + 0.1  * L_temporal
        + 0.001 * L_cov_reg
```

**Files:** `oss/gaussian/train/losses.py`, `tests/gaussian/test_loss.py`

**Term definitions:**
1. **`L_hdr_l1`** — `|tonemap(ŷ) − tonemap(y)|` with the same Reinhard
   tonemap operator OSSFx uses (`oss/train/losses_fx.py:_reinhard_tonemap`).
   For LDR datasets (Sintel sigmoid mode) this collapses to plain L1.
2. **`L_ssim`** — `1 - SSIM` on tonemapped images. Uses the same window-size-11
   default as `oss/train/losses_fx.py`.
3. **`L_lpips`** — LPIPS-VGG (already a project dep). Computed at full HR
   resolution; downsample to 256 px for memory if HR > 1080p.
4. **`L_temporal`** — frame-to-frame consistency: warp the previous frame's
   prediction by motion vectors, compute L1 vs current prediction in static
   regions (motion magnitude < 0.5 px). Supplies the "no ghosting" signal.
5. **`L_cov_reg`** — entropy floor on bank softmax weights:
   `max(0, target_entropy − H(softmax(bank_logits)))`. Prevents the network
   from always picking a single bank entry (which would make ablation of
   bank size trivially flat). Target entropy = 0.6 · log(bank_size).

**Steps:**
1. Implement each term as `nn.Module` for easy weighting.
2. Unit-test each on synthetic targets (e.g. identity → loss ≈ 0).
3. Composite forward + backward on a tiny batch — verify gradients flow.

**Verify:** `pytest tests/gaussian/test_loss.py -v`

**Acceptance:**
- L_hdr_l1 vs identity → loss < 1e-6.
- L_ssim of identical images = 0 (within fp32).
- L_lpips returns scalar of finite shape on (B, 3, H, W).
- L_temporal correctly masks moving regions (verified on synthetic 1-px shift).
- L_cov_reg drops to 0 when softmax is uniform.

**Risks:** LPIPS-VGG is heavy at 4K. Mitigation: downsample to 384-px crops
during the LPIPS computation; full-frame is unnecessary for perceptual loss.

**Time:** ~3 days.

---

### T4.5 — Training script (3080 Ti only — Lambda H100 postponed) ✓ basic wiring landed 2026-05-02

> **2026-05-02 status:** v0 trainer wired end-to-end on the 3080 Ti. Lambda H100 path is postponed; v0 MVP must come from local hardware. Cosine LR schedule + warmup + DDP are not yet implemented (see "Open follow-ups" below). CSV log now writes per-step metrics. `--smoke-test` produces the gate signal.

**Goal:** End-to-end training loop runnable on either local 3080 Ti
(reduced batch size for development) or Lambda H100 SXM 80 GB.

**Files:** `oss/gaussian/train/train.py`, `configs/gaussian/standard.yaml`

**Steps:**
1. Hydra-style YAML config (project already uses hydra-core dep). Fields:
   model.* (tier, bank_size, k_per_tile), data.* (dataset weights, crop size),
   loss.* (term weights), optim.* (lr, wd, betas, scheduler), train.*
   (epochs, batch_size, steps_per_epoch, grad_accum), log.* (wandb / TB / csv).
2. Single-GPU training loop using AdamW + cosine LR schedule + 1k-step warmup.
   Grad-clip at 1.0.
3. DDP support behind `--world-size` flag (Lambda H100 80 GB single-node;
   no multi-node needed at our scale).
4. Mixed precision (bf16 on H100) — fall back to fp32 on the 3080 Ti since
   the renderer hits fp16-NaN edge cases at small scale (verified Sprint 1).
5. Auto-resume from `--resume` checkpoint.
6. CSV log of train/val PSNR + SSIM + LPIPS per 100 steps.
7. `--smoke` flag runs 50 steps on a 16-sample subset for CI sanity.

**Verify:**
```bash
source venv-py312/bin/activate
python -m oss.gaussian.train.train --config configs/gaussian/standard.yaml --smoke
```
Smoke run completes in <2 min on 3080 Ti and produces a checkpoint.

**Acceptance:**
- Smoke run produces a valid checkpoint.
- Loss decreases over the 50-step smoke run (sanity).
- DDP launches on 2 GPUs (verified locally with 1 GPU + `WORLD_SIZE=1`).

**Risks:** rasterizer + autograd memory blow-up if grad checkpointing isn't
applied. Mitigation: torch.utils.checkpoint around the per-tile renderer
section; verify peak memory < 60 GB on H100 at 1080p batch=4.

**Time:** ~4 days.

---

### T4.6 — Pretrain on Sintel + TartanAir (Lambda H100)

**Goal:** Train the **standard** tier (channels (16, 24, 32, 48), K=5,
bank_size=16) on the synthetic mix for ~50 epochs. This is the longest
single task in the sprint.

**Steps:**
1. User-authorised Lambda H100 spend (this is the **GO/NO-GO checkpoint** —
   Sprint 4 stops if approval not given).
2. `scripts/lambda/launch_gaussian_train.sh` — provisions H100, syncs repo,
   pulls dataset from S3, kicks off `python -m oss.gaussian.train.train ...`.
3. ~50 epochs × ~12 min/epoch on H100 = ~10 GPU-hours. With buffer for
   restart/recovery: budget **15 H100-hours = ~$45 at $3/h**.
4. Live monitoring via tail of CSV log + periodic `eval_psnr.py` invocation.
5. Save checkpoints every epoch; keep best-PSNR checkpoint as `best.pt`.

**Verify:** `oss/gaussian/train/eval.py --ckpt best.pt --dataset sintel-val`

**Acceptance:**
- Training completes without crashes / OOMs.
- val-PSNR on Sintel ≥ 28 dB (anchor — OSSPico hits ~30 dB; Gaussian doesn't
  need to beat OSSPico yet, just train without diverging).
- Loss curve monotonic-ish (no divergence).

**Risks:** H100 cost overrun if loss diverges and we restart multiple times.
Mitigation: smoke-run on 3080 Ti before any cloud spend; set Lambda
auto-shutdown after 24 h.

**Time:** ~5 days wall-clock (training is ~1 day; iteration + debug ~4 days).

---

### T4.7 — Fine-tune on Cyberpunk + temporal data

**Goal:** Add the temporal consistency term + Cyberpunk input distribution.
Continue training from T4.6's `best.pt` for another ~10 epochs.

**Steps:**
1. Switch dataloader to enable Cyberpunk samples (15% weight).
2. Enable `L_temporal` term (was off during T4.6 to isolate per-frame quality).
3. Lower LR to 1e-5 for fine-tune.
4. ~10 epochs × ~15 min/epoch = ~3 H100-hours = ~$10.

**Verify:** `eval_psnr.py` + visual diff on a 30-frame Cyberpunk clip.

**Acceptance:**
- val-PSNR drop on Sintel < 0.3 dB (graceful — we're trading Sintel
  performance for temporal stability).
- Temporal-consistency metric (frame-to-frame Δ in flat regions) improves
  ≥ 30% vs T4.6 checkpoint.

**Time:** ~3 days.

---

### T4.8 — Baseline evaluation vs OSSPico

**Goal:** Reproducible head-to-head Gaussian vs OSSPico on a fixed Sintel
test set. This becomes the seed of the graduation criterion (Sprint 7).

**Files:** `oss/gaussian/bench/eval_vs_pico.py`,
`results/gaussian/sprint4_pico_vs_gaussian.csv`

**Steps:**
1. Fixed eval split (`oss/gaussian/bench/sintel_test_split.txt`).
2. Run OSSPico inference on the split, save PSNR/SSIM/LPIPS per frame.
3. Run Gaussian inference (network → decode → renderer at 2× upscale).
4. Run paired t-test on PSNR per frame; report p-value + Cohen's d.
5. Side-by-side video (`scripts/make_compare_video.py`) for subjective check.

**Verify:** `python -m oss.gaussian.bench.eval_vs_pico --output results/gaussian/`

**Acceptance:**
- CSV produced with per-frame metrics for both models.
- Mean PSNR delta reported. **Either direction is acceptable** at this stage —
  we're establishing the baseline, not graduating.

**Risks:** Unfair comparison if scale_factor differs. Lock both to 2.0 here.

**Time:** ~2 days.

---

### T4.9 — Ablation: Bank size 8 vs 16 vs 32

**Goal:** Empirically pick the best bank size. Each variant is a separate
short training run starting from the T4.6 checkpoint with a re-initialised
bank (we cannot warm-start the bank when its dimension changes).

**Files:** `oss/gaussian/train/ablate.py` (driver),
`results/gaussian/sprint4_bank_ablation.csv`

**Steps:**
1. Re-train at each bank size for ~5 epochs (small budget — we're picking
   the working point, not exhaustively training).
2. Report PSNR/SSIM/LPIPS for each bank size on Sintel-val.
3. Also report "bank entropy" — high entropy = network using the bank
   broadly; low = collapsed to a few entries.
4. ~3 H100-hours/variant × 3 variants = 9 H100-hours = ~$27.

**Acceptance:** clear monotonic or U-shape result. If bank size 32 wins by
> 0.3 dB PSNR, adopt it as default for downstream sprints; otherwise stick
with 16 (sprint default).

**Time:** ~2 days.

---

### T4.10 — Ablation: K Gaussians per tile (3, 5, 8)

**Goal:** Pick optimal K. Higher K = more parameters per tile + more renderer
cost; lower K = less expressive output.

**Steps:**
1. Same protocol as T4.9 — re-train standard config at K∈{3, 5, 8}.
2. Report quality + per-frame Gaussian count + render-time impact.
3. ~3 H100-hours/variant × 3 variants = 9 H100-hours = ~$27.

**Acceptance:** report a recommended K as a function of hardware tier
(table in the architecture doc). Pico likely K=3, Standard K=5, Ultra K=8.

**Time:** ~2 days.

---

### T4.11 — Tier scaling fan-out training

**Goal:** Train Pico, Lite, Ultra checkpoints from scratch (Pico/Lite from
the standard checkpoint via channel-pruning warm-start; Ultra from scratch).

**Steps:**
1. Pico — channel-prune the standard model to (8, 16, 24, 32), retrain 10
   epochs.
2. Lite — channel-prune to (16, 24, 32, 40), retrain 10 epochs.
3. Ultra — train from scratch at (24, 32, 48, 64), 50 epochs.
4. ~25 H100-hours total = ~$75.

**Acceptance:** four checkpoints (Pico/Lite/Standard/Ultra) all converge.
Quality vs Gaussian-count Pareto plot in the architecture doc.

**Time:** ~5 days.

---

### T4.12 — TensorRT INT8 export + parity check

**Goal:** Convert PyTorch checkpoint → ONNX → TensorRT INT8 engine.
Calibration on a held-out 256-frame set. Numeric parity within 0.5 dB PSNR.

**Files:** `oss/gaussian/export/trt_export.py`, `trt_calibrator.py`,
`tests/gaussian/test_trt_parity.py`

**Steps:**
1. ONNX export of `GaussianParamNetwork.forward` (head outputs raw tensor —
   the OutputHead decoding stays on the host side, since the renderer is CUDA
   and consumes the decoded params directly).
2. TensorRT 10 build with INT8 + per-channel calibration. Calibrator reads
   the 256-sample subset from disk.
3. Parity test: `pytest tests/gaussian/test_trt_parity.py -v --gpu` —
   compares FP32 PyTorch output against TRT INT8 output on 32 frames.
4. Bench: `python -m oss.gaussian.export.bench_trt` — reports inference time
   on 3080 Ti at 1440p input. Target: < 1.5 ms / frame.

**Verify:** parity test passes (≤ 0.5 dB PSNR delta on rendered output);
bench produces CSV.

**Acceptance:**
- TRT engine built without errors.
- INT8 vs FP32 PSNR delta ≤ 0.5 dB on the parity test.
- Engine inference time within 2× of theoretical bandwidth-bound estimate.

**Risks:** ONNX export of `tile_proj` (kernel-size-equal-to-stride conv) has
been buggy in older opsets — pin opset 17.

**Time:** ~3 days.

---

### T4.13 — Model card + sprint review writeup

**Goal:** Document what shipped so Sprint 5 can consume it.

**Files:**
- `docs/superpowers/gaussian-network-model-card.md` — model card.
- `docs/superpowers/gaussian-network-training-report.md` — training run
  summary + ablation tables + eval-vs-Pico results.
- Update `docs/superpowers/plans/2026-05-01-gaussian-master-plan.md` with
  Sprint 4's actual numbers + any changes to Sprint 5's interface contract.

**Acceptance:** model card includes intended use, training data, compute,
limitations, license note (CC-BY/Apache-only training data). Training
report contains every CSV from T4.6–T4.11 referenced.

**Time:** ~1 day.

---

### T4.14 — Sprint 4 code review checkpoint

**Goal:** Run the code review pipeline on Sprint 4's commits before Sprint 5
starts.

**Steps:**
1. `python -m oss.gaussian.review.run --sprint 4 --commit-range <T4.0 base>..HEAD`.
2. Review artifacts saved to `oss/gaussian/review/artifacts/sprint-4/`.
3. Judge verdict APPROVE → mark sprint complete, proceed to Sprint 5.
4. REQUEST_CHANGES → iterate.
5. BLOCK → escalate to user.

**Verify:** judge verdict file exists and is APPROVE.

**Time:** ~1 day (mostly automated).

---

## Summary timeline (≈ 4 weeks)

| Week | Tasks |
|------|-------|
| 1    | T4.3 (data), T4.4 (loss), T4.5 (training script smoke runs) |
| 2    | T4.6 (Lambda pretrain), T4.7 (fine-tune) |
| 3    | T4.8 (eval), T4.9 (bank ablation), T4.10 (K ablation) |
| 4    | T4.11 (tier fan-out), T4.12 (TRT export), T4.13 (writeup), T4.14 (review) |

## Lambda H100 cost estimate

| Task | H100 hours | $ at $3/h |
|------|-----------:|----------:|
| T4.6 Pretrain        | 15 | $45 |
| T4.7 Fine-tune       |  3 | $10 |
| T4.9 Bank ablation   |  9 | $27 |
| T4.10 K ablation     |  9 | $27 |
| T4.11 Tier fan-out   | 25 | $75 |
| Buffer (debug + restarts) | ~10 | $30 |
| **Total**            | ~71 | **~$210** |

Note: master plan estimated $50–100. The above is the comprehensive estimate
including ablations + tier fan-out. If budget pressure: defer T4.9–T4.11 to
post-Sprint-5 once the canvas integration is verified — that brings the
critical-path cost to ~$60.

## Risk callouts

1. **Renderer + autograd memory** — biggest risk. Mitigation in T4.5 (grad
   checkpoint). Validated on 3080 Ti smoke before any cloud spend.
2. **Cyberpunk frames license** — input distribution only. Sprint 2 hook
   captures must be marked `do_not_use_as_label=True` in the dataset metadata.
3. **Bank collapse** — covariance regulariser term mitigates, but watch
   `bank_entropy` metric every 100 steps. If entropy < 0.3 · log(K) for
   > 1k steps, bump `L_cov_reg` weight 10×.
4. **Tier transfer failure** — channel-prune warm-start may fail. Mitigation:
   if Pico/Lite don't converge from prune, fall back to from-scratch
   training (adds ~10 H100-hours).
5. **TensorRT INT8 quality regression** — common at this network size.
   Mitigation: per-channel calibration, smoothquant if needed, budget
   another day in T4.12 for fallback FP16 engine if INT8 quality unacceptable.
6. **Lambda outage / quota** — RunPod is a documented secondary provider
   (`oss/cloud/runpod_client`); fall back if Lambda is unavailable.

---

## Definition of Done for Sprint 4

- [ ] `oss/gaussian/network/` and `oss/gaussian/train/` ship with full
      pytest coverage on CPU.
- [ ] Standard-tier checkpoint (`best.pt`) produces ≥ 28 dB PSNR on Sintel-val.
- [ ] TensorRT INT8 engine matches FP32 within 0.5 dB.
- [ ] Pico/Lite/Standard/Ultra checkpoints all exist.
- [ ] Bank-size + K-per-tile ablations published (CSV + plot).
- [ ] Eval-vs-OSSPico baseline CSV checked in.
- [ ] Model card + training report + architecture doc up to date.
- [ ] Sprint 4 review pipeline returns APPROVE.

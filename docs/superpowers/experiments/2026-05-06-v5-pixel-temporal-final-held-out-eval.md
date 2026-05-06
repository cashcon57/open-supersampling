# 2026-05-06 — v5-pixel-temporal training completion + final held-out eval

**Status:** Training completed at step 80,000 / 80,000. Held-out eval completed
on the second run, after a bug in the eval pathway was found and fixed. This
memo records both the buggy-run results (preserved for forensic value) and the
corrected results.

## Predecessors

- `2026-05-04-v5-pixel-temporal-runbook.md` — training runbook
- `2026-05-04-v5-pixel-temporal-launch-status.md` — launch status notes
- `2026-05-04-v5-pixel-temporal-distshift-bug-postmortem.md` — the
  channel-zero distribution-shift bug from earlier in the warm-start run
- Commit `b2fa647` — v5 model `forward()` channel-zero fix
- Commit `d25b3b9` — conditional `zero_gbuffer_into_backbone` flag
- Commit `d8ef408` — held-out eval channel-zero fix (this memo's bug)

## Question / hypothesis

Does v5-pixel-temporal warm-started from v4-step-385K and trained 80K steps
on TartanAir Easy (minus held-out env `oldtown`) produce a model that:

- beats v4 baseline on the held-out batch, and
- beats bicubic on the held-out batch on both PSNR and LPIPS in ≥95% of pairs, and
- improves temporal stability over v4 by at least 2× (B/A ratio ≤0.5),

per the v5-pixel-temporal design spec's exit-gate criteria.

## Setup

### Training

- Script: `scripts/sr_train_temporal.py`
- Warm-start: `srcnn-prod-v4-lpips/step-00385000.pt`
- Dataset: TartanAir Easy with `--held-out-envs oldtown`
- LR-synth: `EngineAliasedLRSynth` with jitter + TAA blur (no JPEG, σ=0.5)
- Phase schedule: 1→2 at step 10K (backbone unfreeze); 2→3 at step 60K (LR decay)
- Loss: `L1 + 0.1·SSIM + 0.1·LPIPS-VGG + 0.1·temporal-consistency`
- Hardware: 3080 Ti CUDA, bf16 mixed precision
- Steps: 80,000
- Wall-time: ~32h (started 2026-05-04 22:53 CDT, finished 2026-05-06 08:23 CDT)
- Final checkpoint: `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt`
- Final reported `loss = 0.0617` in phase 3

The `zero_gbuffer_into_backbone` flag was True for this run because warm-start
was used (per the conditional logic in `oss/sr/temporal/model.py` introduced in
commit `d25b3b9`). The flag persists in the checkpoint's saved `args` dict.

### Held-out eval

- Script: `scripts/sr_temporal_held_out.py`
- Manifest: `<train-host-data>\checkpoints\v5_held_out_manifest.json` (64 pairs, all from
  TartanAir env `oldtown`, drawn at training launch and frozen)
- Eval LR-synth: same `EngineAliasedLRSynth` config as training
- Baseline: `srcnn-prod-v4-lpips/step-00385000.pt`
- Hardware: 3080 Ti CUDA

## Result — first run (buggy)

The initial eval reported numbers that signaled a bug rather than a model-
quality finding:

| Metric | v4 baseline (A) | v5-temporal (B) | bicubic |
|---|---|---|---|
| PSNR (dB) | 10.768 | 13.858 | 23.909 |
| LPIPS-VGG | 0.6558 | 0.5664 | 0.2945 |
| B beats bicubic on PSNR | 0/64 | 0/64 | n/a |
| B beats bicubic on LPIPS | 0/64 | 0/64 | n/a |
| B beats A on PSNR | n/a | 51/64 | n/a |
| B beats A on LPIPS | n/a | 64/64 | n/a |
| Temporal stability ratio B/A | n/a | 0.500 | n/a |

(Preserved at `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\held_out_results-buggy-run.json` and `heldout-eval-buggy-run.log`.)

The 15+ dB gap between live training PSNR proxy (28-32 dB throughout phase 2/3)
and held-out PSNR (13.858 dB) was inconsistent with content-difficulty alone.
v4 also dropping to 10.768 dB, both models below bicubic, was the signature of
the same distribution-shift bug we already fixed at training time in commit
`b2fa647`. The eval pathway was bypassing both layers of the channel-zero fix.

## Bug

`scripts/sr_temporal_held_out.py` had two issues:

1. **`_load_temporal()` did not read `zero_gbuffer_into_backbone` from the
   saved checkpoint args.** It built `TemporalSRModel(...)` with the
   constructor-default `False`, so the runtime forward pass fed real
   TartanAir G-buffers into a backbone that was trained on zeroed G-buffers.
   The viz daemon (`sr_temporal_inflight_viz.py`) and the stateless ONNX
   wrapper (`stateless_export.py`) already had the legacy fallback (read
   the flag if present, fall back to `bool(saved.warm_start)` for older
   ckpts that predate the flag); the held-out eval did not.
2. **The baseline (v4) call site at line 418 / 422 fed real G-buffers
   directly to `model_baseline(x_t)`.** v4 is a raw SRCNNSimple/RRDB without
   the conditional-zeroing wrapper. Trained on SRGD with zero G-buffers, it
   produces chromatic-dispersion garbage on real ones. The eval calls v4 once
   per frame as both "A baseline" reference AND v5's cold-start `prev_hr` seed
   at t+1, so this corrupted both A's reported numbers AND v5's first-step
   temporal accumulation.

## Fix

Commit `d8ef408`:

- `_load_temporal()` reads `saved.get("zero_gbuffer_into_backbone")` with
  the same legacy fallback used in `stateless_export.py`.
- New `_baseline_input(x)` helper zeros channels 3..end and sets `[:, 6] = 1.0`
  (the SRGD default-up convention for normals[2]). Both baseline call sites
  (`base_out_t`, `base_out_tp1`) wrap their inputs with this helper.

## Result — second run (corrected)

Same checkpoint, same manifest, same hardware, fixed eval pathway:

| Metric | v4 baseline (A) | **v5-temporal (B)** | bicubic | B vs bicubic |
|---|---|---|---|---|
| PSNR (dB) | 11.718 | **25.703** | 23.909 | **+1.794 dB** |
| LPIPS-VGG | 0.6367 | **0.1666** | 0.2945 | **−43.4%** |
| Temporal stability | 0.1176 | **0.03961** | n/a | B/A = **0.337** |
| B beats bicubic on PSNR | n/a | **64/64** | n/a | 100% |
| B beats bicubic on LPIPS | n/a | **64/64** | n/a | 100% |
| B beats bicubic on PSNR AND LPIPS | n/a | **64/64 (100%)** | n/a | spec target ≥95% |
| B beats A on PSNR | n/a | 64/64 | n/a | clean sweep |
| B beats A on LPIPS | n/a | 64/64 | n/a | clean sweep |

(Result file: `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\held_out_results.json`.)

### Spec exit-gate verdict

| Gate criterion | Threshold | Measured | Pass? |
|---|---|---|---|
| B beats bicubic on PSNR AND LPIPS | ≥ 95% | 100% | **✓** |
| Temporal stability B/A ratio | ≤ 0.5 | 0.337 | **✓ (1.5× margin)** |
| B PSNR > A PSNR | > 0 | +13.985 dB | **✓** |
| B LPIPS < A LPIPS | < 0 | −0.4701 (−73.8%) | **✓** |

All gates passed.

## Interpretation

v5-pixel-temporal is the first OSS model that demonstrably beats bicubic on a
held-out TartanAir batch on both PSNR and LPIPS, every pair, with 1.5× margin
on the temporal-stability spec target. The B/A deltas (+13.985 dB PSNR, −73.8%
LPIPS) reflect that v4 alone is broken on TartanAir distribution; v5
specifically recovers from that and adds the temporal head.

For external context: v5's LPIPS of 0.1666 on this TartanAir batch sits in the
same band as DLSS 4's published ~0.17 LPIPS on AAA games. **This is not a
"v5 beats DLSS 4" claim.** TartanAir is content-class friendlier than the
AAA games DLSS is benchmarked on; the number difference is mostly content,
not method. Apples-to-apples comparison vs DLSS / FSR / XeSS requires the S7
DLL-shim infrastructure (unbuilt) that lets all three run on identical input
in identical content. Current conclusion: v5-pixel-temporal is in the
quality band that competitive temporal SR systems occupy. Whether it stays in
that band on AAA content is unmeasured.

v4 baseline's 11.7 dB on this batch is the SRGD-trained-model-on-TartanAir
distribution-shift failure mode and has been documented separately (commit
`b2fa647` post-mortem). It is not v4's native SRGD performance — v4 measured
~30.1 dB / 0.30 LPIPS on its native SRGD held-out batch.

## Followups

- Sintel held-out eval (separate manifest, content-class diversity check) —
  pending the Sintel `training/depth` package being fetched + extracted.
- v6 implementation can proceed using v5-pixel-temporal as the validated
  pixel-track baseline.
- Capture-tool work continues independently.
- v5-Gaussian-temporal staged validation (Stages 0-2, watchdog respawn pending)
  remains an open question. Given v5-pixel-temporal cleared its exit gate
  cleanly, the case for Stage 3 of v5-Gaussian (full 36h training) weakens —
  Stages 0-2 (~7h) as architectural smoke-test before v6 builds on the
  shared Gaussian code paths is still the right insurance, but Stage 3's
  "produce a v5-Gaussian baseline" rationale is largely covered by
  v5-pixel-temporal already serving that baseline role.

## Artifacts

- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt` — final ckpt
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\held_out_results.json` — corrected eval
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\held_out_results-buggy-run.json` — buggy-run forensic copy
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\heldout-eval.log` — corrected-run log
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\heldout-eval-buggy-run.log` — buggy-run log
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log` — full training log
- `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\viz\step-*.png` — in-flight viz strips

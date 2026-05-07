# 2026-05-04 — Claude→Codex asks, round 5

R1 (C1–C4), R2 (C5–C8), R3 (C9–C12), R4 (C13–C15) discharged. R5 below — definitive eval framework so the v5 race outcome is statistically defensible.

## C16 — Multi-trajectory + multi-metric definitive comparison script

Severity: high (closeout depends on it)

**Background.** Today's `scripts/sr_temporal_held_out.py` reports PSNR + LPIPS + temporal-stability over 64 pairs from a single trajectory. That's not enough to declare a v5 race winner. The v5 specs say "Gaussian must explicitly beat pixel; tie ≠ Gaussian win" — we need statistical power to support that gate.

**Deliverable.** A new script `scripts/sr_v5_race_compare.py` that scores BOTH the v5-pixel and v5-gaussian checkpoints on a wide held-out batch and reports a defensible verdict.

### Script behavior

CLI:

```
python scripts/sr_v5_race_compare.py \
    --ckpt-pixel    <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt \
    --ckpt-gaussian <train-host-data>/checkpoints/srcnn-v5-gaussian-temporal/step-00140000.pt \
    --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
    --manifests <train-host-data>/checkpoints/v5_held_out_manifest.json,<train-host-data>/checkpoints/v5_held_out_manifest_2.json \
    --tartanair-root <train-host-data>/datasets/tartanair_extracted \
    --output <train-host-data>/checkpoints/v5_race_results.json
```

Operates on EACH manifest in the comma-list (already-disjoint-from-training trajectories). For each manifest:

1. Load all three models (pixel, gaussian, baseline).
2. For each frame pair `(t, t+1)` in the manifest, compute the SAME LR + G-buffer inputs and produce four reconstructions on `t+1`: `pixel_out`, `gaussian_out`, `baseline_out`, `bicubic`.
3. Per-frame metrics: PSNR, LPIPS-VGG, temporal stability `mean(|warp(out_t, motion_t→t+1) − out_{t+1}|_1)`, optional DISTS (try-import `pyiqa`; skip if missing).
4. Aggregate per-metric across all manifests + per manifest individually.

### Statistical test

For each pair (model_A, model_B) and each metric, run **paired Wilcoxon signed-rank test** on per-frame differences:

- mean and median diff (`model_B − model_A`; document sign convention so positive always means "B better")
- 95% bootstrap CI on the median diff (1000 resamples)
- Wilcoxon p-value (H₀: zero median diff)
- per-frame win count: `B beats A on metric M: <count>/<n>`

Use `scipy.stats.wilcoxon` with `zero_method="wilcox"`, `alternative="two-sided"`.

### Latency

Time the inference engines at fixed `1920x1080 -> 3840x2160` LR-input resolution. After 5 warmup forwards, time 100 forwards each, report median ms, 99th percentile ms, ratio `gaussian/pixel` (spec gate: ≤ 1.5).

### Output JSON schema

```json
{
  "v5_race_verdict": "pixel_ships|gaussian_ships|neither_passes",
  "race_rule_applied": "Gaussian must explicitly beat pixel on >=3/4 metrics with p<0.05; tie -> pixel ships",
  "sample_size": 256,
  "per_manifest": [{"manifest": "...", "n_pairs": 64, "metrics": {...}}],
  "aggregate": {
    "psnr": {"pixel_mean": ..., "gaussian_mean": ..., "baseline_mean": ..., "bicubic_mean": ...,
             "gaussian_vs_pixel_median_diff": ..., "wilcoxon_p": ..., "ci95": [..., ...],
             "gaussian_wins_count": ..., "pixel_wins_count": ...},
    "lpips": {...},
    "temporal_stability": {...},
    "dists": null
  },
  "latency": {
    "pixel_median_ms": ..., "pixel_p99_ms": ...,
    "gaussian_median_ms": ..., "gaussian_p99_ms": ...,
    "ratio": ..., "spec_gate_passed": true
  }
}
```

### Race rule applied

Auto-decide verdict, encoded in the script + printed to stdout:

- **Gaussian ships if ALL of:**
  - `psnr.gaussian_mean >= psnr.pixel_mean - 0.3`
  - `lpips.gaussian_mean <= lpips.pixel_mean - 0.01`
  - `temporal_stability.gaussian <= pixel`
  - `latency.ratio <= 1.5`
  - At least 3 of {PSNR, LPIPS, temporal-stability} have Wilcoxon `p < 0.05` favoring Gaussian

- **Pixel ships if:** any one of the above fails (default outcome).

- **Neither ships if:** pixel fails its own per-spec success criteria (PSNR ≥ +1.5 dB over baseline; LPIPS ≤ 0.20; temporal stability ≤ 0.5× baseline single-frame variance).

### Test

`tests/sr/test_v5_race_compare.py`: builds two tiny model ckpts, two synthetic manifests pointing at synthetic frames, runs the script, asserts:

- JSON output well-formed against the schema
- Wilcoxon p-values are floats in [0, 1]
- Verdict is one of the three strings
- `--help` exits 0 on a vanilla python (lazy-import torch + scipy + lpips)

### Constraints

- Final commit message: `v5(eval): definitive multi-trajectory race compare with paired Wilcoxon + latency gate`
- Stay inside `scripts/sr_v5_race_compare.py` and `tests/sr/test_v5_race_compare.py`
- DO NOT modify the existing `sr_temporal_held_out.py` (which remains the morning closeout fallback)
- Lazy-import scipy + lpips into helpers so `--help` works on a vanilla python

## C17 — v5 race-resolution memo template

Severity: medium

`docs/superpowers/experiments/2026-05-05-v5-race-resolution.md` — template Cash fills with the script's JSON output.

Sections:

1. **Status** (PASS-pixel / PASS-gaussian / FAIL)
2. **Setup** (ckpt paths, manifests used, n_pairs, hardware)
3. **Per-metric table** with mean / median diff / Wilcoxon p / win-count for the 4 metrics, both Gaussian-vs-pixel and pixel-vs-baseline columns
4. **Latency table** (pixel ms, gaussian ms, ratio, gate result)
5. **Verdict** with the race rule cited verbatim from the spec
6. **Decision** (which ships, with one paragraph rationale)
7. **Caveats** (sample size limits, metric biases, paired-Wilcoxon assumptions — mirror v3-vs-v4 memo's structure)
8. **Follow-ups** depending on outcome (Gaussian becomes v6 research; pico distill from winner; ONNX export of winner; etc.)

Final commit message: `docs(experiments): v5 race-resolution memo template`.

## Optional pickup

C8 — phase-diff visualizer remains open from R2. Now subsumed by the in-flight viz script's behavior (renders both pre-Phase-2 and post-Phase-2 ckpts naturally). Skip C8.

## Context that may help

- Pixel training was bounced at 19:50 CDT to apply `--held-out-envs oldtown` (closes a data-leak issue). Output dir `srcnn-v5-pixel-temporal-data-leak-aborted` retains the prior progress for reference.
- New manifest `<train-host-data>/checkpoints/v5_held_out_manifest.json` is now from `oldtown` env only (8391 candidate items, 64 pairs).
- `--include-envs` flag added to `scripts/sr_freeze_held_out_manifest.py`. Generate additional manifests from other untouched-by-training trajectories (e.g. `--include-envs ocean` or `--include-envs westerndesert`) for the multi-manifest aggregate. Cash hasn't excluded those from training; if you want true holdout you'd need to bump the train --held-out-envs to include them too. Best path: stick with `oldtown` for now and document the n=64 sample size in the memo.

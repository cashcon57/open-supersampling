# 2026-05-04 — v5-pixel-temporal training run, sixth launch (current, fast path)

**Companion to:** `2026-05-04-v5-pixel-temporal-launch-status.md` (history of attempts 1–5).

## Why a sixth launch

Earlier post-resume run (PID 2360, attempt 5) sustained ~32–48 steps/min with GPU at 0% utilization across multiple samples — DataLoader was starving the model on TartanAir's 600 GB dataset (cold OS cache, only 2 worker processes per the hardcoded `num_workers=2` in `scripts/sr_train_temporal.py:236-247`).

Cash's call: "(a) bounce" — restart with `num_workers=4`.

## Fix

Commit `f3930a0` adds:

- `--num-workers <int>` CLI arg, default 2 for back-compat
- `persistent_workers=True` so workers don't tear down between epoch boundaries (TartanAir's effective epoch is huge; persistent workers amortize the per-trajectory startup)

Verified locally: `pytest tests/sr/temporal/test_train_smoke.py -v` → 3/3 PASS post-edit.

## Active process

| Field | Value |
|---|---|
| Launched | 2026-05-04 18:07 CDT |
| Cmd PID | **21192** (orphan-spawn under WMI) |
| Python PID | **10532** |
| Args | `--num-workers 4 --warm-start srcnn-prod-v4-lpips/step-00385000.pt --tartanair-root <train-host-data>\datasets\tartanair_extracted --max-steps 80000 --device cuda` (no `--sintel-root` — Phase 3 falls back to TartanAir until Sintel Depth is wired in) |
| Auto-resume from | `step-00002000.pt` (lost ~25 min of work) |
| Log | `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log` |
| Dashboard | `http://<tailnet-ip>:8080/` (PID 14952) |

Latest Codex process check at 18:33 CDT:

- WMI wrapper/cmd PID `21192` alive.
- Child `python.exe` PID `10532` alive.
- Train log reached step `6160` with finite Phase-1 losses.

Latest Codex process check at 18:49 CDT:

- Phase 2 active.
- GPU util: 100%, memory `12011 / 12288 MiB`.
- Train log reached step `10140`.

## Throughput observations

| Time | Step | Notes |
|---|---|---|
| 18:07 | 2000 | resume |
| 18:08:54 | 2020 | first 20 steps in 90s — workers warming up |
| 18:09:10 | 2060 | 40 steps in 16s = **150 steps/min** |
| 18:30 | 5500 | 3500 steps in 23 min = **152 steps/min average** |
| 18:30 last 60s | 5440→5500 | **300 steps/min instantaneous** |

GPU util: ~50% during steps (was 0% pre-bounce).

## ETA

- Optimistic (sustained 300 steps/min): finish at **~22:30 CDT tonight**
- Realistic (sustained 150 steps/min): finish at **~03:00 CDT tomorrow**
- Either way: morning closeout window is intact

## Phase milestones (from launch-time 18:07)

- Phase 1 → Phase 2 transition at step **10000**: ~18:55 CDT (~50 min in)
- Phase 2 → Phase 3 transition at step **60000**: ~00:25 CDT
- DONE at step **80000**: ~22:30–03:00 CDT depending on rate

## Loss trajectory (pre-Phase-2)

| Step | Loss | t_l1 | tp1_l1 |
|---|---|---|---|
| 5440 | 0.71 | 0.27 | 0.27 |
| 5460 | 0.54 | 0.19 | 0.22 |
| 5480 | 0.47 | 0.15 | 0.17 |
| 5500 | 0.59 | 0.19 | 0.24 |

Healthy decay; Phase 1 (backbone frozen, head + gate only) is fitting the TartanAir HR distribution. Real test comes at Phase 2 transition when backbone unfreezes + temporal-consistency loss activates.

## Sintel Depth — companion download

Cash's URL `https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip` is the live one (the older `files.is.tue.mpg.de/sintel/...` returned 404).

- Downloaded 1.49 GB to `<train-host-data>\datasets\sintel-depth.zip` at 18:25 CDT (8m 23s)
- Currently extracting via `tar -xf` to `<train-host-data>\datasets\sintel-depth-extracted\` (PID 13620)
- Once extracted: needs to be merged into `<train-host-data>\datasets\sintel\training\depth\` (or `--sintel-root` pointed at the new layout) so `SintelGaussianDataset` finds it

## Held-out manifest (canonical, frozen)

Generated on remote: `<train-host-data>\checkpoints\v5_held_out_manifest.json` (64 pairs, seed=0, lr_scale=2.0, manifest_version=1).

A few "non-consecutive frame indices" warnings during generation — expected after the corrupt-flow lazy-skip patches dropped some pairs. Manifest is sound; pairs are valid.

Codex C9 landed manifest consumption in `scripts/sr_temporal_held_out.py` (`a472851`). Morning TartanAir eval should use:

```powershell
python scripts/sr_temporal_held_out.py `
  --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt `
  --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt `
  --tartanair-root <train-host-data>/datasets/tartanair_extracted `
  --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json `
  --n-samples 64
```

Codex C13 added the Sintel manifest and dual-manifest eval support:

- Repo manifest: `docs/superpowers/experiments/v5_held_out_manifest_sintel.json`
- Remote eval copy: `<train-host-data>\checkpoints\v5_held_out_manifest_sintel.json`
- Validation on remote: `dataset_kind=sintel`, `len(SintelGaussianDataset)=1041`, `len(SequentialPairDataset)=1018`, `64` manifest pairs resolved.

Morning dual-manifest eval should use:

```powershell
python scripts/sr_temporal_held_out.py `
  --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt `
  --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt `
  --tartanair-root <train-host-data>/datasets/tartanair_extracted `
  --sintel-root <train-host-data>/datasets/sintel `
  --manifest <train-host-data>/checkpoints/v5_held_out_manifest.json,<train-host-data>/checkpoints/v5_held_out_manifest_sintel.json `
  --n-samples 64
```

## Phase-2 transition observed

Observed by Codex at 18:49 CDT.

- Transition line count: exactly `1`.
- Transition line: `2026-05-04 18:46:11,006 ... phase transition: 1 -> 2  (lr=1.00e-05, backbone_frozen=False)`.
- Phase-1 checkpoint at step `10000` wrote `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00010000.pt`.
- LPIPS initialized immediately after the transition (torchvision/LPIPS warnings appeared), and Phase-2 rows include `tc=...`.

Loss samples around the transition:

| Step | Phase | Loss | t_l1 | tp1_l1 | tc |
|---|---:|---:|---:|---:|---:|
| 9900 | 1 | 0.2830 | 0.0895 | 0.1017 | - |
| 9920 | 1 | 0.6556 | 0.2132 | 0.3063 | - |
| 9940 | 1 | 0.5391 | 0.1659 | 0.2355 | - |
| 9960 | 1 | 0.7637 | 0.2932 | 0.3039 | - |
| 9980 | 1 | 0.4813 | 0.1629 | 0.1820 | - |
| 10000 | 1 | 0.3922 | 0.1216 | 0.1352 | - |
| 10020 | 2 | 0.6598 | 0.1985 | 0.1947 | 0.2679 |
| 10040 | 2 | 0.8852 | 0.2498 | 0.3219 | 0.4264 |
| 10060 | 2 | 0.6329 | 0.1630 | 0.1751 | 0.3255 |
| 10080 | 2 | 0.6286 | 0.1554 | 0.1719 | 0.2247 |
| 10100 | 2 | 0.6380 | 0.1588 | 0.1857 | 0.2822 |
| 10120 | 2 | 0.4332 | 0.0944 | 0.1121 | 0.1313 |
| 10140 | 2 | 0.6805 | 0.1916 | 0.2020 | 0.6283 |

Interpretation:

- No sustained >2x loss spike relative to the pre-transition band. Step `10040` is above 2x the final Phase-1 row, but within the recent Phase-1 variance envelope and settles immediately.
- Throughput dropped from roughly 290 steps/min near step 10000 to roughly 55-65 steps/min in early Phase 2. GPU was at 100% utilization, so this looks compute-bound from LPIPS + unfrozen backbone rather than DataLoader starvation.
- The text log does not print LPIPS component values; Codex filed this as a low-severity logging finding in the rolling review.

## Kill switches (unchanged from r1)

- Stop training: `ssh <train-host> "Stop-Process -Id 21192 -Force"`
- Stop dashboard: `ssh <train-host> "Stop-Process -Id 14952 -Force"`
- Auto-resume picks up from latest `step-XXXXX.pt` if process dies.

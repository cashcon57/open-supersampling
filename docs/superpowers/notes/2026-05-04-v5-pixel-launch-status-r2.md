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
| Cmd PID | (orphan-spawn under WMI) |
| Python PID | **21192** |
| Args | `--num-workers 4 --warm-start srcnn-prod-v4-lpips/step-00385000.pt --tartanair-root <train-host-data>\datasets\tartanair_extracted --max-steps 80000 --device cuda` (no `--sintel-root` — Phase 3 falls back to TartanAir until Sintel Depth is wired in) |
| Auto-resume from | `step-00002000.pt` (lost ~25 min of work) |
| Log | `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log` |
| Dashboard | `http://<tailnet-ip>:8080/` (PID 14952) |

Latest Codex process check at 18:33 CDT:

- WMI wrapper/cmd PID `21192` alive.
- Child `python.exe` PID `10532` alive.
- Train log reached step `6160` with finite Phase-1 losses.

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

## Kill switches (unchanged from r1)

- Stop training: `ssh <train-host> "Stop-Process -Id 21192 -Force"`
- Stop dashboard: `ssh <train-host> "Stop-Process -Id 14952 -Force"`
- Auto-resume picks up from latest `step-XXXXX.pt` if process dies.

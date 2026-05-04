# 2026-05-04 — v5-pixel-temporal training run LAUNCHED

**Launched:** 2026-05-04 16:57 CDT by Claude (controller), per Cash directive "get as much done as possible in the next hour."

## Pre-flight verified

- Local v5 suite: `pytest tests/sr/temporal/ tests/sr/gaussian_temporal/ -v` → 87 passed (commit `c1bad69`)
- TartanAir extraction DONE — `=== TartanAir extraction DONE ===` at 16:13:49 (all 72 zips)
- GPU pre-launch: 841 MiB used / 11244 MiB free / 0% util
- Disk: 1132 GiB free on E:
- CUDA smoke (pixel + Gaussian): both passed in <2s
- v0.2-dev pulled to remote `<train-host-data>/oss-gaussian` (stash applied; `pre-v5-pull-stash-2026-05-04` saved on remote)

## Launch attempts (5th succeeded)

1. PID 16540 — output dir didn't exist → process exited immediately (cmd-line stdout redirect failed). Fixed by `New-Item -Force` on `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal`.
2. PID 8348 — Sintel dataset crashed: `(frame, depth, flow)` triples missing — standard Sintel download has no `depth/` subdir; that's the separate "Sintel Depth" download. Fixed by dropping `--sintel-root` from launch (Phase 3 falls back to TartanAir; full v5 polish on Sintel deferred until Sintel-Depth is fetched).
3. PID 21068 — Windows DataLoader spawn worker-transport failure on `adapt_tartanair`'s local closure. Fixed by `96dad76` (top-level callable classes) + Cash's complementary collate fix `4238915` for `GaussianTrainingExample` dataclass items.
4. cmd PID 27732 / python PID 17256 — crashed at step 260 on a corrupt TartanAir flow `.npy` (`cannot reshape array of size 90040 into shape (480,640,2)`). Initial fix `b8b08c5` skipped bad npy triples by pre-scanning headers, but that made startup too slow at full TartanAir scale.
5. **cmd PID 15652 / python PID 2360 — RUNNING.** Relaunched from `10e75df`, which skips unreadable frame pairs lazily inside `SequentialPairDataset`. Confirmed past the previous crash point: step 340 by 17:23 CDT, loss finite.

## Live training process — python PID 2360

`scripts/sr_train_temporal.py` orphan-spawned via WMI at 17:20 CDT.

Args (Sintel deliberately omitted; see attempt 2 above):

```
--output-dir   <train-host-data>\checkpoints\srcnn-v5-pixel-temporal
--warm-start   <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt
--tartanair-root <train-host-data>\datasets\tartanair_extracted
--max-steps    80000
--device       cuda
```

Log: `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log`

Live snapshot (verified 17:23 CDT, past previous crash point):

```
v5-pixel-temporal: device=cuda tier=standard steps=80000 batch=4 smoke=False warmup=10000 joint_end=60000 lr=1.00e-04
warm-start backbone from <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt
model params: total=626450
phase transition: -1 -> 1  (lr=1.00e-04, backbone_frozen=True)
step=220 phase=1 loss=0.9963 t_l1=0.4141 tp1_l1=0.4266
step=240 phase=1 loss=2.9116 t_l1=1.1811 tp1_l1=1.5499
step=260 phase=1 loss=0.8018 t_l1=0.3072 tp1_l1=0.3337
step=280 phase=1 loss=1.1680 t_l1=0.4972 tp1_l1=0.4923
step=300 phase=1 loss=1.1115 t_l1=0.4713 tp1_l1=0.4625
step=320 phase=1 loss=1.1641 t_l1=0.4901 tp1_l1=0.5080
step=340 phase=1 loss=0.8655 t_l1=0.3410 tp1_l1=0.3555
```

Latest live snapshot (verified 17:35 CDT):

```
python PID 2360 alive; CPU=271.703125; StartTime=5/4/2026 5:20:42 PM
step=1680 phase=1 loss=1.2627 t_l1=0.5337 tp1_l1=0.5646
step=1700 phase=1 loss=1.6662 t_l1=0.7122 tp1_l1=0.7753
step=1720 phase=1 loss=0.9053 t_l1=0.3508 tp1_l1=0.3793
step=1740 phase=1 loss=0.7940 t_l1=0.2951 tp1_l1=0.3397
step=1760 phase=1 loss=0.9259 t_l1=0.3707 tp1_l1=0.3798
step=1780 phase=1 loss=0.9011 t_l1=0.3374 tp1_l1=0.3940
step=1800 phase=1 loss=0.5728 t_l1=0.2107 tp1_l1=0.2321
step=1820 phase=1 loss=0.8555 t_l1=0.3281 tp1_l1=0.3742
```

Loss bouncing around 1–10 — expected for Phase 1 with backbone frozen + temporal head warming up on TartanAir HR distribution (different from SRGD that v4 trained on). Should stabilize as Phase 1 progresses; Phase 2 (10K steps in) unfreezes backbone and adds temporal-consistency loss.

Throughput: ~90 steps/min ≈ 5400 steps/hour. ETA: 80,000 / 5400 ≈ **14.8 hours → finish ~07:54 CDT 2026-05-05**.

GPU utilization: 75%, 3347 MiB / 11244 MiB free. Headroom for the eventual Gaussian run.

### Dashboard PID 14952

`scripts/training_dashboard.py` on port 8080.

- URL: `http://<tailnet-ip>:8080/` (Tailscale alias of `<train-host>`)
- Output dir: `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal`
- Log file: `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/train.log`

## Schedule + expectations

- Phase 1 (steps 0–10K, ~1.5–2 h): backbone frozen, head + gate only, appearance loss (no temporal-consistency yet)
- Phase 2 (10K–60K, ~7–10 h): backbone unfrozen at LR×0.1, full loss with temporal-consistency
- Phase 3 (60K–80K, ~2–3 h): Sintel-only fine-tune at LR×0.01

Total expected: 12–16 h on RTX 3080 Ti. Rolling `metrics.json` written every 2K steps. Auto-resume on process death.

## Sequential GPU directive

**Gaussian training NOT launched yet.** Per Cash's "Sequential unless overlap is safe; test overlap first." The Gaussian runbook's pre-flight gate (3-step check at `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md` §4) must pass before launch — that requires either the pixel run completing OR an explicit overlap-test decision from Cash.

## When you get back, Cash

1. Open `http://<tailnet-ip>:8080/` to monitor dashboard.
2. `ssh <train-host> "Get-Content <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/train.log -Tail 30"` for log tail.
3. `ssh <train-host> "Get-Process -Id 2360"` to confirm process still alive.
4. When Phase 1 completes (~10K steps, ~1.5–2 h from launch), check that PSNR is moving + no NaN.
5. When training completes (~12–16 h):
   - Run held-out eval after Sintel Depth is fetched, or use this TartanAir-only interim command: `python scripts/sr_temporal_held_out.py --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt --tartanair-root <train-host-data>/datasets/tartanair_extracted --n-samples 64`
   - Fill in `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md` with results, rename to actual date.
   - If success-criteria pass: launch Gaussian training via `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`.
   - If not: pixel becomes v5; Gaussian becomes v6+ research.

## Kill switches

- Stop training: `ssh <train-host> "Stop-Process -Id 2360 -Force"`
- Stop dashboard: `ssh <train-host> "Stop-Process -Id 14952 -Force"`
- Auto-resume picks up from latest `step-XXXXX.pt` if process dies.

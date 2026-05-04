# 2026-05-04 — v5-pixel-temporal training run LAUNCHED

**Launched:** 2026-05-04 16:57 CDT by Claude (controller), per Cash directive "get as much done as possible in the next hour."

## Pre-flight verified

- Local v5 suite: `pytest tests/sr/temporal/ tests/sr/gaussian_temporal/ -v` → 87 passed (commit `c1bad69`)
- TartanAir extraction DONE — `=== TartanAir extraction DONE ===` at 16:13:49 (all 72 zips)
- GPU pre-launch: 841 MiB used / 11244 MiB free / 0% util
- Disk: 1132 GiB free on E:
- CUDA smoke (pixel + Gaussian): both passed in <2s
- v0.2-dev pulled to remote `<train-host-data>/oss-gaussian` (stash applied; `pre-v5-pull-stash-2026-05-04` saved on remote)

## Active processes on <train-host>

### Training PID 8348

`scripts/sr_train_temporal.py` orphan-spawned via WMI.

Args:

```
--output-dir   <train-host-data>\checkpoints\srcnn-v5-pixel-temporal
--warm-start   <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt
--tartanair-root <train-host-data>\datasets\tartanair_extracted
--sintel-root  <train-host-data>\datasets\sintel
--max-steps    80000
--device       cuda
```

Log: `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log`

Initial log (verified 16:57:09):

```
v5-pixel-temporal: device=cuda tier=standard steps=80000 batch=4 smoke=False warmup=10000 joint_end=60000 lr=1.00e-04
warm-start backbone from <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt
model params: total=626450
phase transition: -1 -> 1  (lr=1.00e-04, backbone_frozen=True)
```

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
3. `ssh <train-host> "Get-Process -Id 8348"` to confirm process still alive.
4. When Phase 1 completes (~10K steps, ~1.5–2 h from launch), check that PSNR is moving + no NaN.
5. When training completes (~12–16 h):
   - Run held-out eval: `python scripts/sr_temporal_held_out.py --ckpt-temporal <train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt --ckpt-baseline <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt --tartanair-root <train-host-data>/datasets/tartanair_extracted --sintel-root <train-host-data>/datasets/sintel --n-samples 64`
   - Fill in `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out-template.md` with results, rename to actual date.
   - If success-criteria pass: launch Gaussian training via `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`.
   - If not: pixel becomes v5; Gaussian becomes v6+ research.

## Kill switches

- Stop training: `ssh <train-host> "Stop-Process -Id 8348 -Force"`
- Stop dashboard: `ssh <train-host> "Stop-Process -Id 14952 -Force"`
- Auto-resume picks up from latest `step-XXXXX.pt` if process dies.

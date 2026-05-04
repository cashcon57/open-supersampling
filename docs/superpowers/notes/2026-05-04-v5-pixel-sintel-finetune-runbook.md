# v5-pixel-temporal Sintel Fine-Tune Runbook

**Purpose:** Optional post-eval polish run after the TartanAir-only v5-pixel training completes.

**Target host:** `<train-host>`
**Repo path:** `<train-host-data>\oss-gaussian`
**Output dir:** `<train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune`

## Key Constraint

`scripts/sr_train_temporal.py --warm-start` is v4-backbone-only. It calls `TemporalSRModel.load_v4_warm_start()` and expects a checkpoint with `sr_model`. A completed v5 checkpoint has `temporal_model`, so do not pass the v5 checkpoint via `--warm-start`.

For a v5-to-Sintel continuation, stage the completed v5 checkpoint into the new output directory as `step-00080000.pt` and let the trainer's auto-resume path load it.

## Pre-Flight

Run these on `<train-host>`:

```powershell
cd <train-host-data>\oss-gaussian
git fetch origin
git checkout v0.2-dev
git pull --ff-only
git rev-parse HEAD

<windows-home>\Miniconda3\envs\image-gs\python.exe --version
Test-Path <train-host-data>\datasets\sintel
Test-Path <train-host-data>\datasets\sintel\training\depth
Test-Path <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt
```

All `Test-Path` checks must return `True`.

Optional loader smoke:

```powershell
<windows-home>\Miniconda3\envs\image-gs\python.exe - <<'PY'
from pathlib import Path
from oss.gaussian.data import SintelGaussianDataset
from oss.sr.temporal import SequentialPairDataset, adapt_sintel
ds = adapt_sintel(SintelGaussianDataset(root=Path("<train-host-data>/datasets/sintel"), scale=2.0, pass_name="clean"))
pairs = SequentialPairDataset(ds)
print("sintel frames", len(ds), "pairs", len(pairs))
assert len(pairs) > 0
PY
```

## Stage Checkpoint

```powershell
New-Item -ItemType Directory -Force -Path <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune | Out-Null
Copy-Item `
  <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt `
  <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune\step-00080000.pt
```

Do not copy old `metrics.json` or `score_log.json`; this run should write fresh logs.

## Launch

This resumes from step `80000`, then runs steps `80001..100000`. With `--warmup-steps 80000 --joint-end 80000`, the first new step is Phase 3, so the loader selects Sintel and the LR multiplier is `0.01`.

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && set PYTHONUNBUFFERED=1 && <windows-home>\Miniconda3\envs\image-gs\python.exe -u scripts\sr_train_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune --sintel-root <train-host-data>\datasets\sintel --max-steps 100000 --warmup-steps 80000 --joint-end 80000 --lr 1e-6 --device cuda --num-workers 4 > <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune\train.log 2>&1'
}
```

Expected runtime: about 2-3 hours for 20K Sintel-only steps on the RTX 3080 Ti.

## Monitor

```powershell
Get-Content <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune\train.log -Tail 30 -Wait
Get-ChildItem <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune\step-*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Expected early log shape:

- `auto-resume: loading ... step-00080000.pt`
- `resumed at step=80000`
- phase transition to `3` on step `80001`
- log rows use `phase=3`

## Post-Completion Eval

Run the same held-out eval used for the pre-fine-tune checkpoint, then compare the numbers:

```powershell
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_held_out.py `
  --ckpt-temporal <train-host-data>\checkpoints\srcnn-v5-pixel-sintel-finetune\step-00100000.pt `
  --ckpt-baseline <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt `
  --tartanair-root <train-host-data>\datasets\tartanair_extracted `
  --sintel-root <train-host-data>\datasets\sintel `
  --manifest <train-host-data>\checkpoints\v5_held_out_manifest.json,<train-host-data>\checkpoints\v5_held_out_manifest_sintel.json `
  --n-samples 64
```

Compare against the pre-fine-tune v5-pixel-temporal eval. Treat this as a polish run: reject it if Sintel improves but TartanAir regresses enough to threaten the v5 ship gate.

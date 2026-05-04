# v5-pixel-temporal Remote Launch Runbook

**Target host:** `<train-host>` (RTX 3080 Ti, Windows + Miniconda)
**Branch:** `v0.2-dev`
**Companion memo:** [`../experiments/2026-05-04-v5-pixel-temporal-train-start.md`](../experiments/2026-05-04-v5-pixel-temporal-train-start.md)

This is a literal copy-pasteable shell sequence. Run each block in order. Do not skip pre-flight; the smoke run is what catches a broken environment before 12+ hours of GPU time are wasted.

---

## Pre-flight

### 1. SSH in and verify the conda env

```powershell
ssh <train-host>
<windows-home>\Miniconda3\envs\image-gs\python.exe --version
```

Expected: `Python 3.10.x` (or whatever the env was pinned to in v4).

### 2. Pull the repo

```powershell
cd <train-host-data>\oss-gaussian
git fetch origin
git checkout v0.2-dev
git pull --ff-only
git rev-parse HEAD
```

Record the commit SHA in the train-start memo if it drifted from local.

### 3. Verify dataset paths

```powershell
Test-Path <train-host-data>\datasets\tartanair_extracted
Test-Path <train-host-data>\datasets\sintel
```

Both must return `True`. If either is `False`, stop — do not attempt to launch.

### 4. Verify warm-start checkpoint hash

```powershell
Get-FileHash <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt -Algorithm SHA256
```

Expected: `8C079615E6ED2580E21615AB677F16C9B646FB00B74C507617F70B1F6691BEF9`. Mismatch = abort.

### 5. Make the output directory

```powershell
New-Item -ItemType Directory -Force -Path <train-host-data>\checkpoints\srcnn-v5-pixel-temporal | Out-Null
```

### 6. Smoke run on CUDA (5 steps, synthetic-OK)

```powershell
cd <train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_temporal.py --smoke --device cuda --max-steps 5
```

Must exit 0 with no Python tracebacks. If it crashes, fix on `v0.2-dev` and re-pull on remote — do not patch in place.

---

## Launch (orphan-spawn so SSH disconnect can't kill it)

This is the production launch. Orphan-spawn via WMI (`Invoke-CimMethod`) is the pattern that survived the v4 long runs — the spawned process is reparented to the WMI service, so closing the SSH session has no effect.

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-pixel-temporal --warm-start <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt --tartanair-root <train-host-data>\datasets\tartanair_extracted --sintel-root <train-host-data>\datasets\sintel --max-steps 80000 > <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log 2>&1'
}
```

The CIM call returns a `ReturnValue` of `0` and a `ProcessId`. Record the `ProcessId` so you can confirm the run is alive later.

### Verify the orphan is alive

```powershell
Get-Process -Id <ProcessId>
```

Or by name:

```powershell
Get-Process python | Where-Object { $_.Path -like "*image-gs*" }
```

---

## Monitoring

### Tail the log live

```powershell
Get-Content <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log -Tail 20 -Wait
```

`Ctrl-C` to detach — this only kills the tail, not the orphan training process.

### Watch checkpoint cadence

```powershell
Get-ChildItem <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### Restart the dashboard pointing at the new run dir

```powershell
# Kill any prior dashboard on the same port (default 8080)
Get-Process | Where-Object { $_.ProcessName -eq 'python' -and $_.MainWindowTitle -like '*dashboard*' } | Stop-Process

cd <train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_dashboard.py --run-dir <train-host-data>\checkpoints\srcnn-v5-pixel-temporal
```

Browse to `http://<train-host>:8080/` (or whatever port the dashboard prints).

---

## Aborting (if needed)

```powershell
Stop-Process -Id <ProcessId> -Force
```

Then archive the partial run:

```powershell
Rename-Item <train-host-data>\checkpoints\srcnn-v5-pixel-temporal <train-host-data>\checkpoints\srcnn-v5-pixel-temporal-aborted-$(Get-Date -Format 'yyyyMMdd-HHmm')
```

---

## After completion

1. Confirm the final step `step-00080000.pt` exists in `<train-host-data>\checkpoints\srcnn-v5-pixel-temporal\`.
2. Run held-out eval per Plan Task 10:
   ```powershell
   <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_held_out.py `
       --ckpt-temporal <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt `
       --ckpt-baseline <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt `
       --tartanair-root <train-host-data>\datasets\tartanair_extracted `
       --sintel-root <train-host-data>\datasets\sintel `
       --n-samples 64
   ```
3. Fill in `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out.md` with the captured numbers and a written conclusion (pass / fail + reason).
4. If the four success-criteria boxes pass, update the README S5 row per Plan Task 10 Step 3.

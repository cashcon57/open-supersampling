# v5-gaussian-temporal Remote Launch Runbook

**Target host:** `<train-host>` (RTX 3080 Ti, Windows + Miniconda)
**Branch:** `v0.2-dev`
**Companion memo:** [`../experiments/2026-05-04-v5-gaussian-temporal-train-start.md`](../experiments/2026-05-04-v5-gaussian-temporal-train-start.md)
**Sibling track runbook (must finish/idle first by default):** [`./2026-05-04-v5-pixel-temporal-runbook.md`](./2026-05-04-v5-pixel-temporal-runbook.md)

This is a literal copy-pasteable shell sequence. Run each block in order. Do not skip pre-flight; the smoke run is what catches a broken environment before 24+ hours of GPU time are wasted. Do not skip the GPU-share gate; the pixel-temporal run shares this 3080 Ti and OOMs if both run cold without verification.

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

### 4. GPU-share gate (CRITICAL — pixel run must be done or idle)

The remote 3080 Ti is shared with the v5-pixel-temporal control track. Cash directive: **"Sequential GPU train unless overlap is safe; test overlap first."** Do not skip this section.

#### 4a. Confirm pixel-temporal training is complete OR has stopped writing checkpoints

Final pixel checkpoint should be `step-00080000.pt`. If present, the pixel run is done:

```powershell
Test-Path <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt
Get-ChildItem <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-*.pt |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 Name, LastWriteTime
```

Required: either `True` for the final checkpoint, OR the most recent `LastWriteTime` is **older than 30 minutes** (i.e. the pixel run has stopped writing checkpoints — completed, aborted, or hung). If neither condition holds, the pixel run is still active: **abort the Gaussian launch and wait, or escalate to Cash for an explicit overlap go-ahead.**

#### 4b. Confirm no live pixel training Python process

```powershell
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*image-gs*" } |
    Select-Object Id, StartTime, Path, CPU
```

If this returns a row whose `Path` is the `image-gs` python AND it has been running for many hours, that is the pixel orphan still training. Do NOT proceed to launch unless Cash has explicitly cleared overlap. (An empty result is the expected "all clear" state.)

#### 4c. Confirm the GPU itself has headroom

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

Expected for clean sequential launch: `memory.used` well below `memory.total` (rule of thumb: ≤ 2 GB used) and `utilization.gpu` near 0 %. If `memory.used` is more than half of total or `utilization.gpu` is sustained > 20 %, something is still running on the card — go back to 4a/4b and identify it before launching.

If overlap is the deliberate plan (Cash-approved): record the residual `memory.used` and `utilization.gpu` numbers in the train-start memo so post-hoc throughput can be attributed correctly.

### 5. Make the output directory

```powershell
New-Item -ItemType Directory -Force -Path <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal | Out-Null
```

### 6. Smoke run on CUDA (5 steps, synthetic-OK)

```powershell
cd <train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_gaussian_temporal.py --smoke --device cuda --max-steps 5
```

Must exit 0 with no Python tracebacks. If it crashes, fix on `v0.2-dev` and re-pull on remote — do not patch in place.

---

## Launch (orphan-spawn so SSH disconnect can't kill it)

This is the production launch. Orphan-spawn via WMI (`Invoke-CimMethod`) is the pattern that survived the v4 long runs — the spawned process is reparented to the WMI service, so closing the SSH session has no effect.

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_gaussian_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal --tartanair-root <train-host-data>\datasets\tartanair_extracted --sintel-root <train-host-data>\datasets\sintel --max-steps 140000 > <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\train.log 2>&1'
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
Get-Content <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\train.log -Tail 20 -Wait
```

`Ctrl-C` to detach — this only kills the tail, not the orphan training process.

### Watch checkpoint cadence

```powershell
Get-ChildItem <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\step-*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### Watch GPU utilization periodically (24–48 h run; sanity-check a few times a day)

```powershell
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv
```

### Restart the dashboard pointing at the new run dir

```powershell
# Kill any prior dashboard on the same port (default 8080)
Get-Process | Where-Object { $_.ProcessName -eq 'python' -and $_.MainWindowTitle -like '*dashboard*' } | Stop-Process

cd <train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\training_dashboard.py --output-dir <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal --log-file <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\train.log --port 8080 --host 0.0.0.0
```

Browse to `http://<tailnet-ip>:8080/` (Tailscale alias of `<train-host>`).

---

## Aborting (if needed)

```powershell
Stop-Process -Id <ProcessId> -Force
```

Then archive the partial run:

```powershell
Rename-Item <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal-aborted-$(Get-Date -Format 'yyyyMMdd-HHmm')
```

---

## After completion

1. Confirm the final step `step-00140000.pt` exists in `<train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\`.
2. Run held-out eval per Plan Task 12:
   ```powershell
   <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_gaussian_temporal_held_out.py `
       --ckpt-gaussian <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\step-00140000.pt `
       --ckpt-pixel    <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt `
       --tartanair-root <train-host-data>\datasets\tartanair_extracted `
       --sintel-root <train-host-data>\datasets\sintel `
       --n-samples 64
   ```
3. Fill in `docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out.md` with the captured numbers and a written conclusion (pass / fail + reason).
4. Then proceed to the Sprint-5 closeout (Plan Task 14) — the pixel-vs-Gaussian comparison memo and ship decision. Do **not** edit the README S5 row from this runbook; that change happens only in the closeout, after the comparison memo is signed off.

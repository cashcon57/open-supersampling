# Held-out evaluation supervisor.
#
# Runs sr_temporal_held_out.py on every NEW checkpoint that lands in the
# active run's directory and appends the result to score_log.json (which
# the dashboard polls). Idempotent: skips checkpoints whose step is
# already present in score_log.json.
#
# Active run is resolved from RUN_CONFIG via build_public_dashboard.py
# --print-active-run, so rotating runs only requires editing RUN_CONFIG.
#
# Cadence: scans every 60s. The eval itself takes ~2-4 min on the 3080 Ti
# at n-samples=64 with manifest replay; with the trainer writing a ckpt
# every 500 steps (~8-12 min real), one full eval comfortably fits per
# checkpoint cycle.

$pyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs'
$repo = 'E:\oss-gaussian-server'
$ckptRoot = 'E:\checkpoints'
$logFile = 'E:\logs\heldout-eval-supervisor.log'

# v4 baseline ckpt -- pinned. The eval needs a v4 single-frame baseline to
# score against. srcnn-prod-v4-lpips ships in the curated allow-list.
$baselineCkpt = "$ckptRoot\srcnn-prod-v4-lpips\step-00385000.pt"
$manifest = "$ckptRoot\v5_held_out_manifest.json"
$tartanairRoot = 'E:\datasets\tartanair_extracted'
$nSamples = 64

function ResolveActiveRun {
    $name = (& "$pyEnv\python.exe" "$repo\scripts\build_public_dashboard.py" --print-active-run 2>$null) -as [string]
    if ($name) { return $name.Trim() }
    return $null
}

function StepsAlreadyEvaluated {
    param([string]$scoreLog)
    # score_log.json is a JSON ARRAY of dashboard rows (deduped by step in
    # _append_dashboard_score_row). Parse the whole file once and pull
    # every step. Reading line-by-line as JSONL is wrong and silently
    # returns @() -- which made every supervisor restart redundantly
    # re-evaluate every ckpt from scratch.
    if (-not (Test-Path $scoreLog)) { return ,@() }
    try {
        $raw = Get-Content $scoreLog -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($raw)) { return ,@() }
        # PowerShell quirk: `@(ConvertFrom-Json $arr)` and `@($x | ConvertFrom-Json)`
        # both COLLAPSE the resulting Object[] into a single-element array
        # whose [0] is the original Object[]. Direct assignment preserves
        # the array. We then check IEnumerable to handle both array (many
        # rows) and scalar (single-row legacy) cases robustly.
        $payload = ConvertFrom-Json $raw -ErrorAction Stop
        $steps = @()
        if ($payload -is [System.Collections.IEnumerable] -and -not ($payload -is [string])) {
            foreach ($row in $payload) {
                if ($row -ne $null -and $row.step -ne $null) { $steps += [int]$row.step }
            }
        } elseif ($payload -ne $null -and $payload.step -ne $null) {
            $steps += [int]$payload.step
        }
        # Comma-return prevents pipeline unrolling at the function boundary
        # so a single-element array doesn't degenerate into a scalar.
        return ,$steps
    } catch {
        return ,@()
    }
}

function StepFromCkptName {
    param([string]$name)
    if ($name -match 'step-(\d+)\.pt$') { return [int]$matches[1] }
    return -1
}

function Log {
    param([string]$msg)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $logFile -Value "[$stamp] $msg"
}

# File-based GPU mutex shared with heldout-frames-backfill.ps1. Prevents the
# two scripts from racing the non-atomic check-then-spawn on Win32_Process
# and starting two simultaneous evals that thrash the trainer for VRAM.
$script:GpuLockPath = 'C:\temp\oss-heldout-eval.lock'
$script:GpuLockStaleSec = 7200  # 2h: longer than any realistic single eval.

function AcquireGpuLock {
    # Stale-lock recovery: if the existing file is older than $GpuLockStaleSec,
    # treat it as orphaned and remove it. Otherwise New-Item -ErrorAction Stop
    # fails atomically when the file already exists -- which is the lock.
    if (Test-Path $script:GpuLockPath) {
        $existing = Get-Item $script:GpuLockPath -ErrorAction SilentlyContinue
        if ($existing -and ((Get-Date) - $existing.LastWriteTime).TotalSeconds -gt $script:GpuLockStaleSec) {
            Log "removing stale GPU lock from $($existing.LastWriteTime)"
            Remove-Item $script:GpuLockPath -Force -ErrorAction SilentlyContinue
        }
    }
    try {
        $null = New-Item -ItemType File -Path $script:GpuLockPath -Value "$PID" -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function ReleaseGpuLock {
    Remove-Item $script:GpuLockPath -Force -ErrorAction SilentlyContinue
}

Log "supervisor starting"

while ($true) {
    $activeRun = ResolveActiveRun
    if (-not $activeRun) {
        Log "no active run resolved; sleeping"
        Start-Sleep -Seconds 60
        continue
    }
    $runDir = "$ckptRoot\$activeRun"
    if (-not (Test-Path $runDir)) {
        Log "active=$activeRun but $runDir missing; sleeping"
        Start-Sleep -Seconds 60
        continue
    }

    $scoreLog = "$runDir\score_log.json"
    $evaluated = @(StepsAlreadyEvaluated -scoreLog $scoreLog)

    $ckpts = Get-ChildItem $runDir -Filter "step-*.pt" -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Where-Object {
            $s = StepFromCkptName -name $_.Name
            $s -gt 0 -and ($evaluated -notcontains $s)
        }

    if (-not $ckpts) {
        Start-Sleep -Seconds 60
        continue
    }

    # Skip eval while the trainer is currently writing a ckpt (avoid races
    # on partially-flushed files). Heuristic: skip if the file is < 60s old.
    $now = Get-Date
    $ready = $ckpts | Where-Object { ($now - $_.LastWriteTime).TotalSeconds -ge 60 }
    if (-not $ready) {
        Start-Sleep -Seconds 30
        continue
    }

    # GPU mutex acquired around the whole inner loop. Both supervisor and
    # backfill use the same lock file, so only one of them runs an eval at
    # a time. Falling through here means another holder has the lock --
    # back off and re-check next outer loop iteration.
    if (-not (AcquireGpuLock)) {
        Start-Sleep -Seconds 30
        continue
    }
    try {
    foreach ($ck in $ready) {
        $step = StepFromCkptName -name $ck.Name
        # Per-step frame dump dir for the held-out video player. Eval will
        # write 4 streams (model, gt, bicubic, baseline) of 64 PNGs each.
        # GT/bicubic/baseline are skipped after the first eval that wrote
        # them, so only "model" grows per step.
        $framesDir = "$runDir\heldout-frames\step-$($step.ToString('D8'))"
        Log "starting eval: run=$activeRun step=$step ckpt=$($ck.FullName) frames=$framesDir"
        $args = @(
            "scripts\sr_temporal_held_out.py",
            "--ckpt-temporal", "$($ck.FullName)",
            "--ckpt-baseline", "$baselineCkpt",
            "--tartanair-root", "$tartanairRoot",
            "--manifest", "$manifest",
            "--score-log", "$scoreLog",
            "--write-frames-to", "$framesDir",
            "--n-samples", "$nSamples",
            "--device", "cuda"
        ) -join ' '
        $cmdLine = "cmd /c `"cd /d $repo && $pyEnv\python.exe $args >> E:\logs\heldout-eval-$activeRun.log 2>&1`""
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine }
        if ($r.ReturnValue -ne 0) {
            Log "spawn FAILED step=$step rc=$($r.ReturnValue)"
            Start-Sleep -Seconds 30
            continue
        }
        Log "spawned eval pid=$($r.ProcessId) step=$step"
        # Wait for THIS eval to finish before starting the next checkpoint.
        # CIM query failures fail closed: we keep waiting on transient
        # errors instead of declaring completion.
        $running = $true
        while ($running) {
            Start-Sleep -Seconds 15
            try {
                $still = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)" -ErrorAction Stop
                $running = [bool]$still
            } catch {
                Log "CIM query transient failure waiting on pid=$($r.ProcessId); continuing wait"
                # Stay in the wait loop on the assumption that the process
                # is still alive -- safer than racing the GPU.
            }
        }
        Log "completed eval step=$step"
        # Re-read evaluated set for the next iteration.
        $evaluated = @(StepsAlreadyEvaluated -scoreLog $scoreLog)
    }
    } finally {
        ReleaseGpuLock
    }

    Start-Sleep -Seconds 60
}

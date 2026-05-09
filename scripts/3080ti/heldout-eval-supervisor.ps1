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
    if (-not (Test-Path $scoreLog)) { return @() }
    try {
        $lines = Get-Content $scoreLog -ErrorAction Stop
        $steps = @()
        foreach ($ln in $lines) {
            if ([string]::IsNullOrWhiteSpace($ln)) { continue }
            try {
                $obj = $ln | ConvertFrom-Json
                if ($obj.step -ne $null) { $steps += [int]$obj.step }
            } catch {}
        }
        return $steps
    } catch {
        return @()
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

    # Skip if a held-out eval is already running -- the eval is GPU-heavy
    # and we don't want two competing for VRAM with the trainer.
    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like '*sr_temporal_held_out*' }
    if ($running) {
        Start-Sleep -Seconds 30
        continue
    }

    foreach ($ck in $ready) {
        $step = StepFromCkptName -name $ck.Name
        Log "starting eval: run=$activeRun step=$step ckpt=$($ck.FullName)"
        $args = @(
            "scripts\sr_temporal_held_out.py",
            "--ckpt-temporal", "$($ck.FullName)",
            "--ckpt-baseline", "$baselineCkpt",
            "--tartanair-root", "$tartanairRoot",
            "--manifest", "$manifest",
            "--score-log", "$scoreLog",
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
        # Avoids GPU contention and serializes appends to score_log.json.
        do {
            Start-Sleep -Seconds 15
            $still = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)" -ErrorAction SilentlyContinue
        } while ($still)
        Log "completed eval step=$step"
        # Re-read evaluated set for the next iteration.
        $evaluated = @(StepsAlreadyEvaluated -scoreLog $scoreLog)
    }

    Start-Sleep -Seconds 60
}

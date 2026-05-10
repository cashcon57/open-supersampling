# One-shot backfill: write held-out per-frame PNG dumps for ckpts that
# already have a score_log row but were evaluated before the frame-dump
# feature landed. Runs the eval script WITHOUT --score-log (so we don't
# append duplicate scoring rows) and WITH --write-frames-to.
#
# Idempotent: skips any step whose model/sample-000.png already exists.
# Safe to run while the regular supervisor is also running -- both wait
# for any in-flight eval to finish before starting the next, and both
# guard on the GPU not already being held by an eval.
#
# Usage (on the 3080 Ti):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
#     E:\oss-gaussian-server\scripts\3080ti\heldout-frames-backfill.ps1

$ErrorActionPreference = 'Continue'
Set-Location E:\oss-gaussian-server

$pyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs'
$repo = 'E:\oss-gaussian-server'
$ckptRoot = 'E:\checkpoints'
$baselineCkpt = "$ckptRoot\srcnn-prod-v4-lpips\step-00385000.pt"
$manifest = "$ckptRoot\v5_held_out_manifest.json"
$tartanairRoot = 'E:\datasets\tartanair_extracted'
$nSamples = 64
$logFile = 'E:\logs\heldout-frames-backfill.log'

function ResolveActiveRun {
    $name = (& "$pyEnv\python.exe" "$repo\scripts\build_public_dashboard.py" --print-active-run 2>$null) -as [string]
    if ($name) { return $name.Trim() }
    return $null
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
    Write-Host "[$stamp] $msg"
}

# Shared GPU mutex (same path the supervisor uses). Backfill acquires per
# eval, releases after, so the live supervisor can interleave fresh ckpts
# without thrashing the GPU.
$script:GpuLockPath = 'C:\temp\oss-heldout-eval.lock'
$script:GpuLockStaleSec = 7200

function AcquireGpuLock {
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

$activeRun = ResolveActiveRun
if (-not $activeRun) { Log "no active run; aborting"; exit 1 }
$runDir = "$ckptRoot\$activeRun"
$framesRoot = "$runDir\heldout-frames"
$scoreLog = "$runDir\score_log.json"

Log "backfill starting for run=$activeRun"

# Scoped to scored ckpts: backfill only fills frames for steps that have an
# existing score_log row. New / unscored ckpts are the live supervisor's job
# -- we should not race it on those.
$scoredSteps = @()
if (Test-Path $scoreLog) {
    try {
        $raw = Get-Content $scoreLog -Raw -ErrorAction Stop
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            # See supervisor's StepsAlreadyEvaluated for the @()-collapse
            # bug rationale. Direct assignment + IEnumerable test handles
            # both multi-row arrays and single-row scalars.
            $payload = ConvertFrom-Json $raw -ErrorAction Stop
            if ($payload -is [System.Collections.IEnumerable] -and -not ($payload -is [string])) {
                foreach ($r in $payload) {
                    if ($r -ne $null -and $r.step -ne $null) { $scoredSteps += [int]$r.step }
                }
            } elseif ($payload -ne $null -and $payload.step -ne $null) {
                $scoredSteps += [int]$payload.step
            }
        }
    } catch {
        Log "could not parse score_log: $_"
        exit 0
    }
}
if (-not $scoredSteps -or $scoredSteps.Count -eq 0) {
    Log "no scored steps yet; nothing to backfill"
    exit 0
}
$scoredSet = @{}
foreach ($s in $scoredSteps) { $scoredSet[$s] = $true }

$ckpts = Get-ChildItem $runDir -Filter "step-*.pt" -ErrorAction SilentlyContinue |
    Sort-Object Name |
    Where-Object {
        $s = StepFromCkptName -name $_.Name
        $s -gt 0 -and $scoredSet.ContainsKey($s)
    }
Log "found $($ckpts.Count) scored ckpts to consider"

foreach ($ck in $ckpts) {
    $step = StepFromCkptName -name $ck.Name
    if ($step -le 0) { continue }
    $stepStr = $step.ToString('D8')
    $framesDir = "$framesRoot\step-$stepStr"
    # Eval writer drops a per-loader subdir under $framesDir; idempotency
    # marker must match. tartanair is the only loader pico-002 currently
    # exercises. Multi-loader runs would write multiple subdirs and we'd
    # check each.
    $marker = "$framesDir\tartanair\model\sample-000.png"
    if (Test-Path $marker) {
        Log "skip step=$step (frames present)"
        continue
    }

    # Wait for the shared GPU lock. Same file the supervisor holds, so
    # acquiring here serializes us against fresh-ckpt evals. Re-check the
    # marker after acquiring in case the supervisor wrote frames while we
    # were waiting.
    while (-not (AcquireGpuLock)) {
        Start-Sleep -Seconds 20
    }
    if (Test-Path $marker) {
        Log "skip step=$step (frames written by supervisor while we waited)"
        ReleaseGpuLock
        continue
    }

    Log "backfilling step=$step"
    $args = @(
        "scripts\sr_temporal_held_out.py",
        "--ckpt-temporal", "$($ck.FullName)",
        "--ckpt-baseline", "$baselineCkpt",
        "--tartanair-root", "$tartanairRoot",
        "--manifest", "$manifest",
        "--write-frames-to", "$framesDir",
        "--n-samples", "$nSamples",
        "--device", "cuda"
    ) -join ' '
    $cmdLine = "cmd /c `"cd /d $repo && $pyEnv\python.exe $args >> E:\logs\heldout-frames-backfill-eval.log 2>&1`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine }
    if ($r.ReturnValue -ne 0) {
        Log "spawn FAILED step=$step rc=$($r.ReturnValue)"
        ReleaseGpuLock
        continue
    }
    Log "spawned pid=$($r.ProcessId) step=$step"
    # Fail-closed wait: CIM query failures keep us waiting rather than
    # racing the next eval against an unkilled in-flight one.
    $running = $true
    while ($running) {
        Start-Sleep -Seconds 15
        try {
            $still = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)" -ErrorAction Stop
            $running = [bool]$still
        } catch {
            Log "CIM query transient failure waiting on pid=$($r.ProcessId); continuing wait"
        }
    }
    Log "completed step=$step"
    ReleaseGpuLock
}

Log "backfill done"

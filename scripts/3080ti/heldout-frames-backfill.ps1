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

$activeRun = ResolveActiveRun
if (-not $activeRun) { Log "no active run; aborting"; exit 1 }
$runDir = "$ckptRoot\$activeRun"
$framesRoot = "$runDir\heldout-frames"

Log "backfill starting for run=$activeRun"

$ckpts = Get-ChildItem $runDir -Filter "step-*.pt" -ErrorAction SilentlyContinue | Sort-Object Name
Log "found $($ckpts.Count) ckpts"

foreach ($ck in $ckpts) {
    $step = StepFromCkptName -name $ck.Name
    if ($step -le 0) { continue }
    $stepStr = $step.ToString('D8')
    $framesDir = "$framesRoot\step-$stepStr"
    $marker = "$framesDir\model\sample-000.png"
    if (Test-Path $marker) {
        Log "skip step=$step (frames present)"
        continue
    }

    # Wait for any in-flight eval to clear the GPU.
    while ($true) {
        $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like '*sr_temporal_held_out*' }
        if (-not $running) { break }
        Start-Sleep -Seconds 20
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
        continue
    }
    Log "spawned pid=$($r.ProcessId) step=$step"
    do {
        Start-Sleep -Seconds 15
        $still = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)" -ErrorAction SilentlyContinue
    } while ($still)
    Log "completed step=$step"
}

Log "backfill done"

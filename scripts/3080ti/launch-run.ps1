# Launch a new training run on the 3080 Ti, end-to-end.
#
# Workflow this script automates:
#   1. Verify the run name is registered in RUN_CONFIG (so the dashboard
#      knows about it). RUN_CONFIG is the single source of truth for both
#      the build script and the watcher allow-list, so registering a run
#      there propagates everywhere.
#   2. Verify GPU is free.
#   3. Create the output dir.
#   4. Spawn the trainer via WMI Win32_Process.Create (orphan-spawn so it
#      survives terminal close, ssh disconnect, or supervisor restart).
#   5. Confirm the trainer is alive + first log line is sensible.
#
# Usage (on the 3080 Ti):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
#     E:\oss-gaussian-server\scripts\3080ti\launch-run.ps1 `
#     -RunName srcnn-v6.2-pico-002 `
#     -FusionMode concat -SpawnerMode disocclusion -LatentRank 16 `
#     -MaxSteps 100000 -Backbone hat-tiny

param(
    [Parameter(Mandatory=$true)] [string]$RunName,
    [string]$Backbone = 'hat-tiny',
    [string]$FusionMode = 'concat',
    [string]$SpawnerMode = 'disocclusion',
    [int]$LatentRank = 16,
    [int]$MaxSteps = 100000,
    [int]$BatchSize = 4,
    [int]$PatchSize = 128,
    [int]$GradAccum = 4,
    [int]$TrajectoryLength = 4,
    [double]$BaseLr = 2e-4,
    [int]$WarmupSteps = 20000,
    [int]$T0 = 50000,
    [int]$RasterizerOverlap = 8,
    [int]$NumWorkers = 4,
    [int]$FirstCkptStep = 100,
    [int]$CkptEvery = 500,
    [int]$LogEvery = 20,
    [string]$Repo = 'E:\oss-gaussian-server',
    [string]$PyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs',
    [string]$CheckpointsRoot = 'E:\checkpoints',
    [string]$LogsRoot = 'E:\logs',
    [string]$TartanairRoot = 'E:\datasets\tartanair_extracted'
)

$ErrorActionPreference = 'Stop'
Set-Location $Repo

Write-Host "=== [1/5] verify run is in RUN_CONFIG ==="
$known = & "$PyEnv\python.exe" "$Repo\scripts\build_public_dashboard.py" --print-run-names 2>$null
if ($LASTEXITCODE -ne 0 -or -not $known) {
    Write-Host "  ERROR: could not read RUN_CONFIG via build_public_dashboard.py"
    exit 2
}
$knownLines = $known -split "`r?`n" | Where-Object { $_ }
if ($knownLines -notcontains $RunName) {
    Write-Host "  ERROR: '$RunName' is not registered in RUN_CONFIG."
    Write-Host "  Add it to scripts\build_public_dashboard.py RUN_CONFIG dict, commit, push,"
    Write-Host "  then re-run this script. Known runs:"
    foreach ($k in $knownLines) { Write-Host "    - $k" }
    exit 3
}
Write-Host "  OK ($RunName found)"

Write-Host ""
Write-Host "=== [2/5] check GPU is free ==="
$gpu = nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
$gpuParts = $gpu -split ',' | ForEach-Object { $_.Trim() }
$gpuMemMb = [int]$gpuParts[0]
Write-Host "  current GPU usage: $gpuMemMb MiB / util $($gpuParts[1])%"
if ($gpuMemMb -gt 2000) {
    Write-Host "  WARN: GPU has $gpuMemMb MiB used. Existing process may conflict."
    $existing = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like '*sr_train_v6*' }
    foreach ($p in $existing) {
        Write-Host ("    existing trainer PID {0}: {1}" -f $p.ProcessId, ($p.CommandLine.Substring(0, [Math]::Min(120, $p.CommandLine.Length))))
    }
    Write-Host "  Aborting. Stop the existing run first or unset OSS_FORCE_LAUNCH if you really mean it."
    if (-not $env:OSS_FORCE_LAUNCH) { exit 4 }
}

Write-Host ""
Write-Host "=== [3/5] create output dir ==="
$out = "$CheckpointsRoot\$RunName"
if (Test-Path $out) {
    if ((Get-ChildItem $out | Measure-Object).Count -gt 0 -and -not $env:OSS_FORCE_LAUNCH) {
        Write-Host "  ERROR: $out already has contents. Move/delete it first or set OSS_FORCE_LAUNCH=1."
        exit 5
    }
} else {
    New-Item -ItemType Directory -Path $out | Out-Null
    Write-Host "  created $out"
}

$logFile = "$LogsRoot\$RunName.log"
Write-Host ""
Write-Host "=== [4/5] orphan-spawn trainer via WMI ==="
$cli = @(
    "scripts\sr_train_v6.py",
    "--backbone $Backbone",
    "--fusion-mode $FusionMode",
    "--spawner-mode $SpawnerMode",
    "--latent-rank $LatentRank",
    "--max-steps $MaxSteps",
    "--output-dir $out",
    "--tartanair-root $TartanairRoot",
    "--batch-size $BatchSize",
    "--patch-size $PatchSize",
    "--grad-accum $GradAccum",
    "--trajectory-length $TrajectoryLength",
    "--base-lr $BaseLr",
    "--warmup-steps $WarmupSteps",
    "--T0 $T0",
    "--rasterizer-overlap $RasterizerOverlap",
    "--no-spawn-offset-random",
    "--no-spawn-subpixel-jitter",
    "--num-workers $NumWorkers",
    "--device cuda",
    "--bf16",
    "--first-ckpt-step $FirstCkptStep",
    "--ckpt-every $CkptEvery",
    "--log-every $LogEvery"
) -join ' '
$inner = "cd /d $Repo && $PyEnv\python.exe $cli > `"$logFile`" 2>&1"
$wmiCmd = "cmd /c `"$inner`""
# Hide spawned cmd window so long training runs don't keep a visible console open.
$startupHidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $wmiCmd; ProcessStartupInformation = $startupHidden }
Write-Host ("  WMI ReturnValue=$($result.ReturnValue) PID=$($result.ProcessId)")
if ($result.ReturnValue -ne 0) {
    Write-Host "  spawn FAILED. cmd was:"
    Write-Host "    $wmiCmd"
    exit 6
}

Write-Host ""
Write-Host "=== [5/5] verify trainer alive (waiting 30s) ==="
Start-Sleep -Seconds 30
$pythons = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*sr_train_v6*$RunName*" }
foreach ($p in $pythons) {
    Write-Host ("  trainer PID {0}" -f $p.ProcessId)
}
if (-not $pythons) {
    Write-Host "  ERROR: no trainer process found after 30s. Check $logFile for errors."
    if (Test-Path $logFile) { Get-Content $logFile -Tail 30 }
    exit 7
}

Write-Host ""
Write-Host "=== first 25 log lines ==="
if (Test-Path $logFile) {
    Get-Content $logFile -TotalCount 25
}

Write-Host ""
Write-Host "=== launched. Reminders ==="
Write-Host "  - The watcher's allow-list + GPU-status active run + viz-daemon active run"
Write-Host "    all derive from RUN_CONFIG, so the dashboard should pick this up within"
Write-Host "    one watcher cycle (~30s) once the first metrics row lands."
Write-Host "  - Tail the log: powershell -c 'Get-Content $logFile -Wait -Tail 25'"
Write-Host "  - Stop later: scripts\3080ti\stop-run.ps1 -RunName $RunName"

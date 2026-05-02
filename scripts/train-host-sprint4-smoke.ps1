# Sprint 4 low-capacity smoke test on 3080 Ti.
# Must beat bicubic on >=1 scene to gate Lambda H100 spend.
# Per 2026-05-01 validation decision memo, Decision 1.
# Don't use $ErrorActionPreference = "Stop" — Python sends INFO logging to
# stderr, which PS treats as a NativeCommandError under Stop policy and kills
# the script. We rely on $LASTEXITCODE for the real exit status.
$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logfile = "<train-host-data>\logs\sprint4-smoke-$timestamp.log"
$outdir = "<train-host-data>\checkpoints\sprint4-smoke-$timestamp"
New-Item -ItemType Directory -Force -Path "<train-host-data>\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

# Use SRGD GameEngineData (real game engine renders, paired LR/HR).
# Sintel ships clean+flow on this machine but no depth, so the Sintel adapter
# can't form a (frame, depth, flow) triple. SRGD has no G-buffers and the
# adapter zeros them out — fine for a smoke-test gate against bicubic.
$srgdRoot = "<train-host-data>\datasets\srgd"
if (-not (Test-Path "$srgdRoot\data\GameEngineData\ActionRPG")) {
    Write-Error "SRGD ActionRPG scene missing at $srgdRoot\data\GameEngineData\ActionRPG"
    exit 1
}

# Use the existing image-gs miniconda env — it already has torch 2.4.1 + CUDA wired.
$pythonExe = "<windows-home>\Miniconda3\envs\image-gs\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Python not found at $pythonExe. Run conda env list to verify."
    exit 1
}

cd <train-host-data>\oss-gaussian

$pyArgs = @(
    "-m", "oss.gaussian.train.train",
    "--smoke-test",
    "--dataset", "srgd",
    "--srgd-scene", "ActionRPG",
    "--dataset-root", $srgdRoot,
    "--output-dir", $outdir,
    "--max-steps", "20000",
    "--max-time-seconds", "10800",
    "--eval-every", "500",
    "--device", "cuda"
)

Write-Host "Starting Sprint 4 smoke test -> $logfile"
& $pythonExe @pyArgs 2>&1 | Tee-Object -FilePath $logfile
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "==== SMOKE TEST DONE (exit $exitCode) ===="
Write-Host "Log:    $logfile"
Write-Host "Output: $outdir"
Write-Host ""
Write-Host "Final result line:"
Select-String -Path $logfile -Pattern "SMOKE TEST RESULT" | Select-Object -Last 1

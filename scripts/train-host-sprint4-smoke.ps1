# Sprint 4 low-capacity smoke test on 3080 Ti.
# Must beat bicubic on >=1 scene to gate Lambda H100 spend.
# Per 2026-05-01 validation decision memo, Decision 1.
$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logfile = "<train-host-data>\logs\sprint4-smoke-$timestamp.log"
$outdir = "<train-host-data>\checkpoints\sprint4-smoke-$timestamp"
New-Item -ItemType Directory -Force -Path "<train-host-data>\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

$sintelRoot = "<train-host-data>\datasets"  # expects <train-host-data>\datasets\MPI-Sintel-complete\
if (-not (Test-Path "$sintelRoot\MPI-Sintel-complete\training\clean")) {
    Write-Error "Sintel dataset missing at $sintelRoot\MPI-Sintel-complete\training\clean"
    exit 1
}

# Activate conda env with CUDA + torch installed
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
conda activate oss

cd <windows-home>\open-reconstruction-suite  # adjust path if different

$pyArgs = @(
    "-m", "oss.gaussian.train.train",
    "--smoke-test",
    "--sintel-sequence", "alley_1",
    "--dataset-root", $sintelRoot,
    "--output-dir", $outdir,
    "--max-steps", "20000",
    "--max-time-seconds", "10800",
    "--eval-every", "500",
    "--device", "cuda"
)

Write-Host "Starting Sprint 4 smoke test -> $logfile"
python @pyArgs 2>&1 | Tee-Object -FilePath $logfile
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "==== SMOKE TEST DONE (exit $exitCode) ===="
Write-Host "Log:    $logfile"
Write-Host "Output: $outdir"
Write-Host ""
Write-Host "Final result line:"
Select-String -Path $logfile -Pattern "SMOKE TEST RESULT" | Select-Object -Last 1

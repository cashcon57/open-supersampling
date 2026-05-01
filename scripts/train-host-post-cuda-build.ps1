# OSS-Gaussian Sprint 1 close-out: build gsplat + run CUDA tests + bench.
# Triggered by cuda-watcher.ps1 when CUDA Toolkit appears at the canonical path.
$ErrorActionPreference = 'Continue'
$Log = 'C:\Windows\Temp\oss-gaussian-sprint1-close.log'
$Repo = '<train-host-data>\oss-gaussian'
$Conda = '<windows-home>\Miniconda3\Scripts\conda.exe'

function Log($msg) {
  $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
  Add-Content -Path $Log -Value $line
  Write-Host $line
}

Log "=== START Sprint 1 close-out ==="

# Locate nvcc and add to PATH
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
if (-not (Test-Path $cudaRoot)) {
  Log "ABORT: CUDA toolkit not at $cudaRoot"
  exit 2
}
$ver = Get-ChildItem $cudaRoot -Directory | Select-Object -Last 1
$nvccDir = Join-Path $ver.FullName 'bin'
$env:PATH = "$nvccDir;$env:PATH"
$env:CUDA_PATH = $ver.FullName
$env:CUDA_HOME = $ver.FullName
Log "CUDA found: $($ver.FullName)"
& nvcc --version | Out-File -FilePath $Log -Append

# 1. pip install gsplat from the vendored Image-GS submodule
Log "Building gsplat CUDA extension (this is the slow step, 5-15 min)..."
Push-Location "$Repo\oss\gaussian\renderer\vendor\image_gs\gsplat"
& $Conda run -n image-gs pip install -v -e . *>&1 | Out-File -FilePath $Log -Append
$gsplatExit = $LASTEXITCODE
Pop-Location
Log "gsplat build exit code: $gsplatExit"

if ($gsplatExit -ne 0) {
  Log "gsplat build FAILED. See log above."
  exit 3
}

# 2. Verify gsplat importable
Log "Verifying gsplat imports..."
& $Conda run -n image-gs python -c "from gsplat import rasterize_gaussians_sum; print('gsplat OK')" *>&1 | Out-File -FilePath $Log -Append

# 3. Run all gaussian tests including CUDA paths
Log "Running tests/gaussian/ on CUDA..."
Push-Location $Repo
& $Conda run -n image-gs python -m pytest tests/gaussian/ -v 2>&1 | Out-File -FilePath $Log -Append
$testExit = $LASTEXITCODE
Pop-Location
Log "test exit code: $testExit"

# 4. Run the benchmark
Log "Running renderer benchmark on RTX 3080 Ti..."
Push-Location $Repo
& $Conda run -n image-gs python -m oss.gaussian.renderer.bench 2>&1 | Out-File -FilePath $Log -Append
$benchExit = $LASTEXITCODE
Pop-Location
Log "bench exit code: $benchExit"
if (Test-Path "$Repo\oss\gaussian\renderer\bench") {
  Log "Bench CSV files:"
  Get-ChildItem "$Repo\oss\gaussian\renderer\bench" *.csv | ForEach-Object { Log "  $($_.FullName)" }
}

# 5. Run code review pipeline (dry-run since no API key on the box)
Log "Running review pipeline dry-run on Sprint 1..."
Push-Location $Repo
& $Conda run -n image-gs python -m oss.gaussian.review.run --sprint 1 --commit-range origin/main..HEAD --dry-run 2>&1 | Out-File -FilePath $Log -Append
Pop-Location

Log "=== DONE Sprint 1 close-out ==="
exit 0

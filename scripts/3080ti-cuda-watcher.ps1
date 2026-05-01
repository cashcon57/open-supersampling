# Watcher: polls every 60s for CUDA Toolkit appearance, then triggers Sprint 1 close-out.
$ErrorActionPreference = 'Continue'
$Log = 'C:\Windows\Temp\oss-gaussian-cuda-watcher.log'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
$builder = 'C:\Windows\Temp\post-cuda-build.ps1'
$maxMinutes = 60

function Log($msg) {
  $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
  Add-Content -Path $Log -Value $line
}

Log "=== START watcher; max $maxMinutes min ==="
$start = Get-Date
while ($true) {
  $elapsed = (Get-Date) - $start
  if ($elapsed.TotalMinutes -gt $maxMinutes) {
    Log "Timeout after $maxMinutes min. CUDA still not present."
    exit 1
  }
  if (Test-Path $cudaRoot) {
    $vers = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue
    if ($vers) {
      Log "CUDA detected after $([int]$elapsed.TotalMinutes) min. Triggering builder."
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $builder *>&1 | Out-File -FilePath $Log -Append
      Log "Builder exited with $LASTEXITCODE"
      exit $LASTEXITCODE
    }
  }
  Start-Sleep -Seconds 60
}

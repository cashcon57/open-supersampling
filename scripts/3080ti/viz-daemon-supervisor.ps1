# Wrapper that restarts the inflight viz daemon if it dies. Idempotent --
# checks for an existing healthy daemon first; only respawns if missing.
# The active run is derived from RUN_CONFIG via the build script's
# --print-active-run mode, so the only place to edit when rotating runs is
# scripts/build_public_dashboard.py.
$pyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs'
$repo = 'E:\oss-gaussian-server'

while ($true) {
  # Re-resolve active run on every cycle so rotating it via RUN_CONFIG
  # auto-propagates within ~60s without restarting the supervisor.
  $activeRun = (& "$pyEnv\python.exe" "$repo\scripts\build_public_dashboard.py" --print-active-run 2>$null) -as [string]
  if ($activeRun) { $activeRun = $activeRun.Trim() }
  if (-not $activeRun) { $activeRun = 'srcnn-v6.2-pico-002' }

  # Kill any viz daemons pointed at a stale active run.
  $stale = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'sr_temporal_inflight_viz' -and $_.CommandLine -notmatch [regex]::Escape($activeRun) }
  foreach ($p in $stale) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\logs\viz-supervisor.log' -Value "[$stamp] killing stale viz daemon pid=$($p.ProcessId) (active=$activeRun)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }

  $alive = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "sr_temporal_inflight_viz.*$activeRun" }
  if (-not $alive) {
    $daemonCmd = "cmd /c `"cd /d $repo && $pyEnv\python.exe scripts\sr_temporal_inflight_viz.py --output-dir E:\checkpoints\$activeRun --ckpt-v5 E:\checkpoints\srcnn-v5-pixel-temporal-validated\step-00080000.pt --manifest E:\checkpoints\v5_held_out_manifest.json --primary-version v6 --backbone hat-tiny --tartanair-root E:\datasets\tartanair_extracted --device cpu --interval 60 --n-pairs 2 >> E:\logs\viz-daemon-$activeRun.log 2>&1`""
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\logs\viz-supervisor.log' -Value "[$stamp] daemon missing for $activeRun - respawning"
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $daemonCmd } | Out-Null
  }
  Start-Sleep -Seconds 60
}

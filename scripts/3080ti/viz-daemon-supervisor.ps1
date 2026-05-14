# Wrapper that restarts the inflight viz daemon if it dies. Idempotent --
# checks for an existing healthy daemon first; only respawns if missing.
# The active run is derived from RUN_CONFIG via the build script's
# --print-active-run mode, so the only place to edit when rotating runs is
# scripts/build_public_dashboard.py.
$pyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs'
# Repo path: oss-gaussian is the training repo (oss-gaussian-server was a
# legacy alias that no longer exists; both pointed at the same code).
$repo = 'E:\oss-gaussian'

# Hide spawned daemon's console window: SW_HIDE = 0
$startupHidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }

while ($true) {
  # Re-resolve active run on every cycle. Prefer v7 over v6: --version v7
  # returns the v7 active run, else fall back to default v6. Lets the
  # supervisor flip between architectures without code changes.
  $activeRun = (& "$pyEnv\python.exe" "$repo\scripts\build_public_dashboard.py" --print-active-run --version v7 2>$null) -as [string]
  $primaryVersion = 'v7'
  if ($activeRun) { $activeRun = $activeRun.Trim() }
  if (-not $activeRun) {
    $activeRun = (& "$pyEnv\python.exe" "$repo\scripts\build_public_dashboard.py" --print-active-run 2>$null) -as [string]
    $primaryVersion = 'v6'
    if ($activeRun) { $activeRun = $activeRun.Trim() }
  }
  if (-not $activeRun) {
    # Nothing active anywhere; sleep and retry.
    Start-Sleep -Seconds 60
    continue
  }

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
    # Build the daemon command line. The v7 path takes --ckpt-v6 for the
    # baseline comparison column (v6.2-pico-002 step-00074000.pt), and
    # skips the v5 manifest. The v6 path keeps the legacy v5-ckpt + manifest.
    if ($primaryVersion -eq 'v7') {
      $v62ckpt = 'E:\checkpoints\srcnn-v6.2-pico-002\step-00074000.pt'
      # v7 manifest has 6 oldtown pairs spread across the trajectory
      # so each strip row shows a visually-distinct chunk of the run.
      $manifest = 'E:\checkpoints\v7_held_out_manifest.json'
      $daemonCmd = "cmd /c `"cd /d $repo && $pyEnv\python.exe scripts\sr_temporal_inflight_viz.py --output-dir E:\checkpoints\$activeRun --primary-version v7 --tartanair-root E:\datasets\tartanair_extracted --manifest $manifest --device cpu --interval 120 --n-pairs 6 --ckpt-v6 $v62ckpt >> E:\logs\viz-daemon-$activeRun.log 2>&1`""
    } else {
      $daemonCmd = "cmd /c `"cd /d $repo && $pyEnv\python.exe scripts\sr_temporal_inflight_viz.py --output-dir E:\checkpoints\$activeRun --ckpt-v5 E:\checkpoints\srcnn-v5-pixel-temporal-validated\step-00080000.pt --manifest E:\checkpoints\v5_held_out_manifest.json --primary-version v6 --backbone hat-tiny --tartanair-root E:\datasets\tartanair_extracted --device cpu --interval 60 --n-pairs 2 >> E:\logs\viz-daemon-$activeRun.log 2>&1`""
    }
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\logs\viz-supervisor.log' -Value "[$stamp] daemon missing for $activeRun (primary=$primaryVersion) - respawning"
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $daemonCmd; ProcessStartupInformation = $startupHidden } | Out-Null
  }
  Start-Sleep -Seconds 60
}

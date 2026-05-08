# Wrapper that restarts the inflight viz daemon if it dies. Idempotent --
# checks for an existing healthy daemon first; only respawns if missing.
# Points at the active training run; update the run name + version when the
# active run rotates (current: srcnn-v6.2-pico-002, v6.2 architecture).
$activeRun = 'srcnn-v6.2-pico-002'
$daemonCmd = "cmd /c `"cd /d E:\oss-gaussian-server && C:\Users\cashc\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_inflight_viz.py --output-dir E:\checkpoints\$activeRun --ckpt-v5 E:\checkpoints\srcnn-v5-pixel-temporal-validated\step-00080000.pt --manifest E:\checkpoints\v5_held_out_manifest.json --primary-version v6 --backbone hat-tiny --tartanair-root E:\datasets\tartanair_extracted --device cpu --interval 60 --n-pairs 2 >> E:\logs\viz-daemon-$activeRun.log 2>&1`""

while ($true) {
  $alive = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "sr_temporal_inflight_viz.*$activeRun" }
  if (-not $alive) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\logs\viz-supervisor.log' -Value "[$stamp] daemon missing for $activeRun - respawning"
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $daemonCmd } | Out-Null
  }
  Start-Sleep -Seconds 60
}

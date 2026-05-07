# Wrapper that restarts the inflight viz daemon if it dies. Idempotent —
# checks for an existing healthy daemon first; only respawns if missing.
$daemonCmd = 'cmd /c "cd /d E:\oss-gaussian & C:\Users\cashc\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_inflight_viz.py --output-dir E:\checkpoints\srcnn-v6.1-pico-001 --ckpt-v5 E:\checkpoints\srcnn-v5-pixel-temporal-validated\step-00080000.pt --manifest E:\checkpoints\v5_held_out_manifest.json --primary-version v6 --backbone hat-tiny --tartanair-root E:\datasets\tartanair_extracted --device cpu --interval 60 --n-pairs 2 >> E:\logs\viz-daemon-v6.1.log 2>&1"'

while ($true) {
  $alive = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'sr_temporal_inflight_viz.*srcnn-v6.1-pico-001' }
  if (-not $alive) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\logs\viz-supervisor.log' -Value "[$stamp] daemon missing - respawning"
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $daemonCmd } | Out-Null
  }
  Start-Sleep -Seconds 60
}

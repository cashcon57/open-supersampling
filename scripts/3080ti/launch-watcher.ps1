# launch-watcher.ps1 — WMI-orphan-spawn the watch_and_publish.sh loop on the
# 3080 Ti. Reads from /e/checkpoints (the actual training checkpoints) and
# pushes data.json + viz strips to R2 via the upload Worker.
#
# Eliminates the Mac-side watcher + sync_remote_runs entirely. Mac no longer
# needs to be online for the dashboard to update.
#
# Pre-reqs:
#   - Git for Windows at C:\Program Files\Git\bin\bash.exe
#   - rsync via scoop+cwrsync at /c/Users/cashc/scoop/shims/rsync
#   - Fresh repo at E:\oss-gaussian-server tracking origin/main
#   - .secrets/r2-credentials.env populated at E:\oss-gaussian-server\.secrets\
#   - Python via /c/Users/cashc/Miniconda3/envs/image-gs/python.exe (build_public_dashboard.py)
#
# Run:
#   ssh 3080ti-windows powershell -NoProfile -ExecutionPolicy Bypass `
#     -File C:\Users\cashc\3080ti\launch-watcher.ps1
#
# Tail the log via:
#   ssh 3080ti-windows '& "C:\Program Files\Git\bin\bash.exe" -lc "tail -f /tmp/watch_and_publish.log"'

$bashCmd = 'cd /e/oss-gaussian-server && git pull --ff-only origin main; bash scripts/watch_and_publish.sh >> /tmp/watch_and_publish.log 2>&1'
$cmd = '"C:\Program Files\Git\bin\bash.exe" -lc "' + $bashCmd + '"'
# Hide spawned bash console (SW_HIDE = 0); matches the supervisor pattern.
$startupHidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
  CurrentDirectory = 'E:\oss-gaussian-server'
  ProcessStartupInformation = $startupHidden
}
Write-Output "ProcessId=$($r.ProcessId) ReturnValue=$($r.ReturnValue)"

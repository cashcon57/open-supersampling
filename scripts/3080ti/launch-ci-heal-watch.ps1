# launch-ci-heal-watch.ps1 — WMI-orphan-spawn the ci_auto_heal --watch
# loop on the 3080ti so it survives SSH disconnect.
#
# Pre-reqs:
#   - Git for Windows installed at C:\Program Files\Git
#   - Fresh repo clone at E:\oss-gaussian-server (clean tracking origin/main)
#   - jq.exe on PATH (winget install jqlang.jq)
#   - gh authenticated (gh auth login on Windows native)
#
# Run from PowerShell on the 3080ti, or invoke remotely via:
#   ssh 3080ti-windows powershell -NoProfile -ExecutionPolicy Bypass `
#     -File C:\Users\cashc\3080ti\launch-ci-heal-watch.ps1
#
# Tail the log via:
#   ssh 3080ti-windows '& "C:\Program Files\Git\bin\bash.exe" -lc "tail -f /tmp/ci_auto_heal_watch.log"'

$bashCmd = 'cd /e/oss-gaussian-server && bash scripts/ci_auto_heal.sh --watch >> /tmp/ci_auto_heal_watch.log 2>&1'
$cmd = '"C:\Program Files\Git\bin\bash.exe" -lc "' + $bashCmd + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
  CurrentDirectory = 'E:\oss-gaussian-server'
}
Write-Output "ProcessId=$($r.ProcessId) ReturnValue=$($r.ReturnValue)"

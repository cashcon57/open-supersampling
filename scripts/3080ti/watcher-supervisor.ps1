# watcher-supervisor.ps1 — restart watch_and_publish.sh if it's not running.
#
# Belt-and-suspenders for the dashboard deploy pipeline:
#   * Layer 1: watch_and_publish.sh has a periodic git pull (commit 357460f).
#   * Layer 2 (this script): if the watcher process dies, this supervisor
#     starts a new one. Schedule via Task Scheduler to run every 60 seconds.
#   * Layer 3: GitHub Actions deploy-dashboard.yml ships dashboard files to
#     R2 directly on push to main, completely independent of the 3080 Ti.
#
# Schedule via Task Scheduler:
#   Action: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\cashc\3080ti\watcher-supervisor.ps1
#   Trigger: at logon + repeat every 1 minute indefinitely
#
# The supervisor itself is idempotent (safe to run repeatedly). It checks
# whether bash.exe is running watch_and_publish and, if not, spawns a new
# detached watcher via WMI Win32_Process Create — same orphan-spawn pattern
# the launch-watcher.ps1 script uses.

$ErrorActionPreference = 'Continue'

$logFile = 'C:\temp\watcher-supervisor.log'
function Log {
    param([string]$msg)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$ts $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

# Look for any bash.exe process whose CommandLine contains 'watch_and_publish'.
# Get-CimInstance gives us CommandLine for each Win32_Process row, which is
# what we need (the process name itself is just 'bash.exe').
$running = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'bash.exe' -and
        $_.CommandLine -and
        $_.CommandLine.Contains('watch_and_publish')
    }

if ($running) {
    # Watcher alive; nothing to do.
    Log "OK: watch_and_publish alive (pid=$($running.ProcessId))"
    exit 0
}

# Watcher dead → restart it via the same orphan-spawn pattern as
# launch-watcher.ps1. This decouples the new watcher from this supervisor
# process so the supervisor can exit cleanly.
Log "WARN: watch_and_publish not running; restarting"

$bashCmd = 'cd /e/oss-gaussian-server && git pull --ff-only origin main; bash scripts/watch_and_publish.sh >> /tmp/watch_and_publish.log 2>&1'
$cmd     = '"C:\Program Files\Git\bin\bash.exe" -lc "' + $bashCmd + '"'
$result  = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $cmd
    CurrentDirectory = 'E:\oss-gaussian-server'
}

if ($result.ReturnValue -eq 0) {
    Log "OK: watcher restarted (pid=$($result.ProcessId))"
    exit 0
} else {
    Log "ERROR: watcher restart failed (ReturnValue=$($result.ReturnValue))"
    exit 1
}

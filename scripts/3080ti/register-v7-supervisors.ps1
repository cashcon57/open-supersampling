# Register the v7 held-out eval supervisor as a Windows scheduled task.
#
# Mirrors how the v6 supervisor is registered (Logon + Boot triggers,
# hidden window, runs as the current user). Run this ONCE on the 3080 Ti
# from an elevated pwsh session; verify with `schtasks /Query`.
#
# This script only registers the task; the supervisor itself
# (heldout-eval-v7-supervisor.ps1) takes over from there.

$taskName = 'OSS-HeldOut-Eval-V7-Supervisor'
$supervisorPath = 'E:\oss-gaussian-server\scripts\3080ti\heldout-eval-v7-supervisor.ps1'

if (-not (Test-Path $supervisorPath)) {
    Write-Error "supervisor script not found at $supervisorPath -- aborting registration"
    exit 1
}

# Action: launch pwsh with the supervisor script, window hidden so it
# doesn't pop a console on logon. -NoProfile keeps startup fast and
# avoids profile-induced surprises in the scheduled-task environment.
$action = New-ScheduledTaskAction `
    -Execute 'pwsh.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$supervisorPath`""

# Triggers: at user logon AND at boot. Boot trigger catches headless
# reboots; logon trigger catches manual sign-ins after a power loss.
$triggers = @(
    New-ScheduledTaskTrigger -AtLogOn
    New-ScheduledTaskTrigger -AtStartup
)

# Run as the current user with highest privileges. The 3080 Ti runs as a
# single-user box, so -UserId on $env:USERNAME is correct.
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

# Settings: Hidden=$true keeps the task entry from popping pwsh windows;
# AllowStartIfOnBatteries + DontStopIfGoingOnBatteries keeps it alive on
# laptops; ExecutionTimeLimit 0 = no kill timeout (supervisor is a
# long-lived while-true loop and must not be culled).
$settings = New-ScheduledTaskSettingsSet `
    -Hidden:$true `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

# Re-register idempotently: if the task exists, unregister first.
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Host "task $taskName already registered; unregistering for re-register"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $triggers `
    -Principal $principal `
    -Settings $settings `
    -Description 'OSS v7 held-out eval supervisor (Phase 3 dashboard lane).' | Out-Null

Write-Host "registered scheduled task: $taskName"
Write-Host ''
Write-Host '--- Verification ---'
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State, Triggers
Write-Host ''
Write-Host '--- schtasks /Query ---'
schtasks /Query /TN $taskName /V /FO LIST | Select-String -Pattern 'TaskName|Status|Next Run Time|Last Run Time|Last Result|Run As User'
Write-Host ''
Write-Host 'Done. Supervisor will start at next logon/boot, or run:'
Write-Host "  Start-ScheduledTask -TaskName $taskName"

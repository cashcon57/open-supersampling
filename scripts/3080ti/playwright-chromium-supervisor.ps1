# Restarts the Playwright Chromium CDP endpoint if the tailnet-bound port drops.
$ErrorActionPreference = 'Continue'

$debugAddress = '100.121.175.55'
$debugPort = 9222
$pollSeconds = 60
$logDir = 'E:\logs'
$logPath = Join-Path $logDir 'playwright-chromium-supervisor.log'
$startScript = Join-Path $PSScriptRoot 'start-playwright-chromium.ps1'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log {
  param([Parameter(Mandatory = $true)][string]$Message)
  $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Add-Content -Path $logPath -Value "[$stamp] $Message"
}

function Start-PlaywrightChromium {
  if (-not (Test-Path -LiteralPath $startScript)) {
    Write-Log "start script missing: $startScript"
    return
  }

  try {
    $output = & $startScript 2>&1 | Out-String
    $trimmed = $output.Trim()
    if ($trimmed) {
      Write-Log "spawn requested: $trimmed"
    } else {
      Write-Log 'spawn requested'
    }
  } catch {
    Write-Log "spawn failed: $($_.Exception.Message)"
  }
}

Write-Log "supervisor starting for $debugAddress`:$debugPort"
$lastReachable = $null

while ($true) {
  $reachable = $false

  try {
    $reachable = [bool](Test-NetConnection -ComputerName $debugAddress -Port $debugPort -InformationLevel Quiet -WarningAction SilentlyContinue)
  } catch {
    Write-Log "probe failed: $($_.Exception.Message)"
  }

  if ($reachable) {
    if ($lastReachable -ne $true) {
      Write-Log "cdp reachable at $debugAddress`:$debugPort"
    }
  } else {
    if ($lastReachable -eq $false) {
      Write-Log "cdp still down at $debugAddress`:$debugPort - respawning"
    } else {
      Write-Log "cdp down at $debugAddress`:$debugPort - respawning"
    }
    Start-PlaywrightChromium
  }

  $lastReachable = $reachable
  Start-Sleep -Seconds $pollSeconds
}

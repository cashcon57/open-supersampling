param(
  [string]$CyberpunkExe = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
  [string]$DllPath = "E:\Temp\oss-capture-artifacts\oss_capture.dll",
  [string]$Python = "C:\Users\cashc\Miniconda3\python.exe",
  [string]$Injector = "E:\Temp\inject_dll_once.py",
  [int]$PreInjectWaitSeconds = 10,
  [int]$PostInjectWaitSeconds = 20,
  [int]$InjectionTimeoutSeconds = 10
)

$ErrorActionPreference = "Stop"

function Read-Optional([string]$Path) {
  if (Test-Path -LiteralPath $Path) {
    return @(Get-Content -LiteralPath $Path)
  }
  return @()
}

if (-not (Test-Path -LiteralPath $CyberpunkExe)) {
  throw "Cyberpunk executable not found: $CyberpunkExe"
}
if (-not (Test-Path -LiteralPath $DllPath)) {
  throw "DLL not found: $DllPath"
}

$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$logDir = Join-Path $localAppData "oss-gaussian"
$logPath = Join-Path $logDir "interception.log"
$markerPath = Join-Path $logDir "init-marker.log"
Remove-Item -Force -ErrorAction SilentlyContinue $logPath,$markerPath

$gameDir = Split-Path -Parent $CyberpunkExe
$proc = Start-Process -FilePath $CyberpunkExe -ArgumentList @("--launcher-skip", "-benchmark") -WorkingDirectory $gameDir -PassThru
Start-Sleep -Seconds $PreInjectWaitSeconds

$running = @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue | Select-Object -First 1)
$injectOutput = ""
$injectExit = $null
if ($running.Count -gt 0) {
  $injectOutput = & $Python $Injector --pid $running[0].Id --dll $DllPath --timeout $InjectionTimeoutSeconds 2>&1
  $injectExit = $LASTEXITCODE
  Start-Sleep -Seconds $PostInjectWaitSeconds
}

$modules = @()
foreach ($p in @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)) {
  try {
    $modules += @($p.Modules | Where-Object {
      $_.ModuleName -match "^(oss_|dxgi|d3d12|nvngx|sl\.|winmm|version|wininet|dbghelp)"
    } | Select-Object @{Name="process_id";Expression={$p.Id}}, ModuleName, FileName)
  } catch {
    $modules += [pscustomobject]@{
      process_id = $p.Id
      ModuleName = "<module-enumeration-failed>"
      FileName = $_.Exception.Message
    }
  }
}

Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue |
  Stop-Process -Force -ErrorAction SilentlyContinue

[ordered]@{
  cyberpunk_exe = $CyberpunkExe
  dll_path = $DllPath
  process_id = $proc.Id
  running_before_inject = $running.Count
  inject_exit_code = $injectExit
  inject_output = $injectOutput
  loaded_modules = $modules
  marker = Read-Optional $markerPath
  log_tail = if (Test-Path -LiteralPath $logPath) { @(Get-Content -LiteralPath $logPath -Tail 80) } else { @() }
} | ConvertTo-Json -Depth 8

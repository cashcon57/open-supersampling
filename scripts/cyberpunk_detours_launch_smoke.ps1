param(
  [string]$CyberpunkExe = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
  [string]$DllPath = "E:\Temp\oss-capture-artifacts\oss_capture.dll",
  [string]$WithDll = "E:\Temp\withdll.exe",
  [int]$WaitSeconds = 45,
  [string]$ReportPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CyberpunkExe)) {
  throw "Cyberpunk executable not found: $CyberpunkExe"
}
if (-not (Test-Path -LiteralPath $DllPath)) {
  throw "DLL not found: $DllPath"
}
if (-not (Test-Path -LiteralPath $WithDll)) {
  throw "withdll.exe not found: $WithDll"
}

$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$logDir = Join-Path $localAppData "oss-gaussian"
$logPath = Join-Path $logDir "interception.log"
$markerPath = Join-Path $logDir "init-marker.log"
Remove-Item -Force -ErrorAction SilentlyContinue $logPath,$markerPath

$gameDir = Split-Path -Parent $CyberpunkExe
$exeName = Split-Path -Leaf $CyberpunkExe
$env:OSS_GAUSSIAN_INIT_MARKER = "1"
Push-Location $gameDir
try {
  $withDllOutput = & $WithDll "/d:$DllPath" $exeName "--launcher-skip" "-benchmark" 2>&1
  $withDllExit = $LASTEXITCODE
} finally {
  Pop-Location
  Remove-Item Env:\OSS_GAUSSIAN_INIT_MARKER -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds $WaitSeconds

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

$result = [ordered]@{
  cyberpunk_exe = $CyberpunkExe
  dll_path = $DllPath
  withdll_exit = $withDllExit
  withdll_output = $withDllOutput
  loaded_modules = $modules
  marker = if (Test-Path -LiteralPath $markerPath) { @(Get-Content -LiteralPath $markerPath) } else { @() }
  log_tail = if (Test-Path -LiteralPath $logPath) { @(Get-Content -LiteralPath $logPath -Tail 120) } else { @() }
}

$json = $result | ConvertTo-Json -Depth 8
if ($ReportPath -ne "") {
  $reportDir = Split-Path -Parent $ReportPath
  if ($reportDir -ne "") {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
  }
  $json | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}
$json

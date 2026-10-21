param(
  [string]$CyberpunkExe = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
  [int]$WaitSeconds = 45,
  [string]$ReportPath = ""
)

$ErrorActionPreference = "Stop"

function Get-FileVersionState([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return @{ exists = $false }
  }
  $item = Get-Item -LiteralPath $Path
  $vi = $item.VersionInfo
  return @{
    exists = $true
    path = $item.FullName
    length = $item.Length
    last_write_time = $item.LastWriteTimeUtc.ToString("o")
    company = $vi.CompanyName
    product = $vi.ProductName
    description = $vi.FileDescription
    version = $vi.FileVersion
  }
}

if (-not (Test-Path -LiteralPath $CyberpunkExe)) {
  throw "Cyberpunk executable not found: $CyberpunkExe"
}

$gameDir = Split-Path -Parent $CyberpunkExe
$interestingNames = @(
  "dxgi.dll",
  "nvngx_dlss.dll",
  "nvngx_dlssd.dll",
  "nvngx_dlssg.dll",
  "sl.interposer.dll",
  "sl.dlss.dll",
  "sl.dlss_d.dll",
  "sl.dlss_g.dll",
  "winmm.dll",
  "version.dll",
  "wininet.dll",
  "dbghelp.dll",
  "d3d12.dll",
  "amd_fidelityfx_dx12.dll",
  "ffx_fsr3upscaler_x64.dll",
  "libxess.dll"
)

$inventory = @{}
foreach ($name in $interestingNames) {
  $inventory[$name] = Get-FileVersionState (Join-Path $gameDir $name)
}

$proc = Start-Process -FilePath $CyberpunkExe -ArgumentList @("--launcher-skip", "-benchmark") -WorkingDirectory $gameDir -PassThru
Start-Sleep -Seconds $WaitSeconds

$running = @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)
$modules = @()
foreach ($p in $running) {
  try {
    $modules += @($p.Modules | Where-Object {
      $_.ModuleName -match "^(dxgi|d3d12|nvngx|sl\.|winmm|version|wininet|dbghelp|amd_|ffx_|libxess)"
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
  game_dir = $gameDir
  waited_seconds = $WaitSeconds
  process_id = $proc.Id
  running_count = $running.Count
  inventory = $inventory
  loaded_modules = $modules
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

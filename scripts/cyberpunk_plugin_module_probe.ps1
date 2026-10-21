param(
  [string]$CyberpunkExe = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
  [int]$WaitSeconds = 45
)

$ErrorActionPreference = "Stop"
$gameDir = Split-Path -Parent $CyberpunkExe
$proc = Start-Process -FilePath $CyberpunkExe -ArgumentList @("--launcher-skip", "-benchmark") -WorkingDirectory $gameDir -PassThru
Start-Sleep -Seconds $WaitSeconds

$modules = @()
foreach ($p in @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)) {
  try {
    $modules += @($p.Modules | Where-Object {
      $_.FileName -match "\.asi$|cyber_engine|red4ext|VERSION\.dll|WINMM\.dll|dxgi\.dll|oss_capture"
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
  process_id = $proc.Id
  modules = $modules
} | ConvertTo-Json -Depth 6

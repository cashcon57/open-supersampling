param(
  [string]$GameDir = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64",
  [int]$WaitSeconds = 20
)

$ErrorActionPreference = "Stop"

Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue |
  Stop-Process -Force -ErrorAction SilentlyContinue

$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
  if (-not (Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)) {
    break
  }
  Start-Sleep -Milliseconds 500
}

$dxgiPath = Join-Path $GameDir "dxgi.dll"
$dxgiBackup = Get-ChildItem -LiteralPath $GameDir -Filter "dxgi.dll.oss-smoke-*.bak" |
  Sort-Object LastWriteTimeUtc -Descending |
  Select-Object -First 1
$ngxPath = Join-Path $GameDir "nvngx_dlss.dll"
$ngxBackup = Join-Path $GameDir "nvngx_dlss.dll.oss-backup"

if ($dxgiBackup) {
  Copy-Item -LiteralPath $dxgiBackup.FullName -Destination $dxgiPath -Force
  Remove-Item -LiteralPath $dxgiBackup.FullName -Force
}

if (Test-Path -LiteralPath $ngxBackup) {
  Copy-Item -LiteralPath $ngxBackup -Destination $ngxPath -Force
}

[ordered]@{
  game_dir = $GameDir
  restored_dxgi_from = if ($dxgiBackup) { $dxgiBackup.FullName } else { $null }
  dxgi = if (Test-Path -LiteralPath $dxgiPath) { (Get-Item -LiteralPath $dxgiPath).Length } else { $null }
  nvngx = if (Test-Path -LiteralPath $ngxPath) { (Get-Item -LiteralPath $ngxPath).Length } else { $null }
  remaining_cyberpunk_processes = @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue).Count
} | ConvertTo-Json -Depth 4

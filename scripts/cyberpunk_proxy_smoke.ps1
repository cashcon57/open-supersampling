param(
  [string]$CyberpunkExe = "D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe",
  [string]$ArtifactDir = "E:\Temp\oss-capture-artifacts",
  [string]$LoaderProxyName = "dxgi.dll",
  [switch]$NoLoaderProxy,
  [switch]$NoNgxProxy,
  [switch]$StreamlineProxy,
  [switch]$Fsr3Proxy,
  [switch]$Fsr3UpscalerProxy,
  [switch]$FsrBackendProxy,
  [switch]$FfxApiProxy,
  [switch]$TemporarilyDisableMods,
  [switch]$EnableInitMarker,
  [switch]$InitInDllMain,
  [switch]$LaunchViaSteam,
  [string]$SteamExe = "C:\Program Files (x86)\Steam\steam.exe",
  [string]$SteamAppId = "1091500",
  [string]$CaptureMode = "trickle",
  [ValidateSet("", "DLSS", "FSR2", "FSR3", "FSR4", "XeSS")]
  [string]$ForceResolutionScaling = "",
  [string]$ForceUpscalerQuality = "Quality",
  [int]$WaitSeconds = 75,
  [switch]$SkipCaptureValidation,
  [string]$PythonExe = "python",
  [string]$ReportPath = "",
  [string]$TracePath = "",
  [string[]]$LaunchArgs = @("--launcher-skip", "--intro-skip", "-benchmark")
)

$ErrorActionPreference = "Stop"

function Write-Trace([string]$Message) {
  if ($TracePath -eq "") {
    return
  }
  $traceDir = Split-Path -Parent $TracePath
  if ($traceDir -ne "") {
    New-Item -ItemType Directory -Force -Path $traceDir | Out-Null
  }
  $line = "{0:o} {1}" -f (Get-Date), $Message
  Add-Content -LiteralPath $TracePath -Value $line -Encoding UTF8
}

function Get-FileState([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return @{ exists = $false }
  }
  $item = Get-Item -LiteralPath $Path
  $hash = $null
  try {
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
  } catch {
    $hash = "<hash-failed: $($_.Exception.Message)>"
  }
  return @{
    exists = $true
    length = $item.Length
    last_write_time = $item.LastWriteTimeUtc.ToString("o")
    sha256 = $hash
  }
}

function Read-LogTail([string]$Path, [int]$Lines = 80) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return @()
  }
  try {
    return @(Get-Content -LiteralPath $Path -Tail $Lines)
  } catch {
    return @("<read-failed: $($_.Exception.Message)>")
  }
}

function Invoke-CaptureValidation([string]$PendingRoot) {
  if ($SkipCaptureValidation) {
    return @{ skipped = $true; reason = "SkipCaptureValidation" }
  }
  $validator = Join-Path $PSScriptRoot "validate_capture_samples.py"
  if (-not (Test-Path -LiteralPath $validator)) {
    return @{ skipped = $true; reason = "validator script not found"; script = $validator }
  }
  if (-not (Test-Path -LiteralPath $PendingRoot)) {
    return @{ skipped = $true; reason = "pending root not found"; pending_dir = $PendingRoot }
  }
  $output = @()
  $exitCode = $null
  try {
    $output = @(& $PythonExe $validator $PendingRoot --json 2>&1 | ForEach-Object { [string]$_ })
    $exitCode = $LASTEXITCODE
  } catch {
    return @{
      skipped = $false
      valid = $false
      exit_code = $null
      error = $_.Exception.Message
      output = $output
    }
  }
  $joined = $output -join "`n"
  $parsed = $null
  try {
    if ($joined.Trim() -ne "") {
      $parsed = $joined | ConvertFrom-Json -ErrorAction Stop
    }
  } catch {
    $parsed = $null
  }
  return @{
    skipped = $false
    valid = ($exitCode -eq 0)
    exit_code = $exitCode
    report = $parsed
    output = $output
  }
}

function ConvertTo-JsonReady($Value) {
  if ($null -eq $Value) {
    return $null
  }
  if ($Value -is [string] -or $Value -is [bool] -or
      $Value -is [int] -or $Value -is [long] -or
      $Value -is [double] -or $Value -is [decimal]) {
    return $Value
  }
  if ($Value -is [System.Collections.IDictionary]) {
    $obj = [ordered]@{}
    foreach ($key in $Value.Keys) {
      $obj[[string]$key] = ConvertTo-JsonReady $Value[$key]
    }
    return [pscustomobject]$obj
  }
  if ($Value -is [System.Collections.IEnumerable]) {
    $items = @()
    foreach ($item in $Value) {
      $items += ConvertTo-JsonReady $item
    }
    return $items
  }
  return [string]$Value
}

function ConvertTo-SmokeJson($Value) {
  if ($null -eq $Value) {
    return "null"
  }
  if ($Value -is [bool]) {
    if ($Value) { return "true" }
    return "false"
  }
  if ($Value -is [int] -or $Value -is [long] -or
      $Value -is [double] -or $Value -is [decimal]) {
    return [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $Value)
  }
  if ($Value -is [System.Collections.IDictionary]) {
    $parts = @()
    foreach ($key in $Value.Keys) {
      $parts += (ConvertTo-SmokeJson ([string]$key)) + ":" + (ConvertTo-SmokeJson $Value[$key])
    }
    return "{" + ($parts -join ",") + "}"
  }
  if (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string])) {
    $parts = @()
    foreach ($item in $Value) {
      $parts += ConvertTo-SmokeJson $item
    }
    return "[" + ($parts -join ",") + "]"
  }
  $text = [string]$Value
  $text = $text.Replace("\", "\\")
  $text = $text.Replace('"', '\"')
  $text = $text.Replace("`r", "\r")
  $text = $text.Replace("`n", "\n")
  $text = $text.Replace("`t", "\t")
  return '"' + $text + '"'
}

function Get-CyberpunkLoadedModules([int]$ProcessId) {
  if ($ProcessId -le 0) {
    return @()
  }

  $job = Start-Job -ScriptBlock {
    param([int]$TargetProcessId)
    $loaded = @()
    $p = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if ($null -eq $p) {
      return @()
    }
    $loaded += @($p.Modules | Where-Object {
      $_.ModuleName -match "^(dxgi|d3d12|nvngx|sl\.|winmm|version|wininet|dbghelp|amd_|ffx_|libxess|oss_)"
    } | Select-Object @{Name="process_id";Expression={$p.Id}}, ModuleName, FileName)
    return @($loaded)
  } -ArgumentList $ProcessId

  if (-not (Wait-Job -Job $job -Timeout 8)) {
    Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    return @([pscustomobject]@{
      process_id = $ProcessId
      ModuleName = "<module-enumeration-timeout>"
      FileName = "Timed out after 8 seconds"
    })
  }

  try {
    $items = @(Receive-Job -Job $job -ErrorAction Stop)
    return @($items | ForEach-Object {
      [ordered]@{
        process_id = [int]$_.process_id
        module_name = [string]$_.ModuleName
        file_name = [string]$_.FileName
      }
    })
  } catch {
    return @([pscustomobject]@{
      process_id = $ProcessId
      ModuleName = "<module-enumeration-failed>"
      FileName = $_.Exception.Message
    })
  } finally {
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
  }
}

function Wait-NoCyberpunkProcess([int]$TimeoutSeconds = 20) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)) {
      return $true
    }
    Start-Sleep -Milliseconds 500
  }
  return -not (Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)
}

function Disable-ModFileForSmoke([string]$Path, [string]$Stamp) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return $null
  }
  $backup = "$Path.oss-disabled-$Stamp"
  if (Test-Path -LiteralPath $backup) {
    throw "Refusing to disable mod file because backup already exists: $backup"
  }
  Move-Item -LiteralPath $Path -Destination $backup -Force
  return [ordered]@{
    path = $Path
    backup = $backup
    before = Get-FileState $backup
  }
}

function Clear-Or-Refuse-StaleBackup([string]$SourcePath, [string]$BackupPath) {
  if (-not (Test-Path -LiteralPath $BackupPath)) {
    return $false
  }
  if (-not (Test-Path -LiteralPath $SourcePath)) {
    throw "Refusing to run smoke: backup exists but source is missing. source=$SourcePath backup=$BackupPath"
  }
  $sourceHash = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
  $backupHash = (Get-FileHash -LiteralPath $BackupPath -Algorithm SHA256).Hash
  if ($sourceHash -ne $backupHash) {
    throw "Refusing to run smoke: stale backup differs from current source. source=$SourcePath backup=$BackupPath"
  }
  Remove-Item -LiteralPath $BackupPath -Force
  return $true
}

function Get-CyberpunkUserSettingIndex([string]$Name, [object]$Value) {
  $text = [string]$Value
  $key = $text
  if ($text -eq "XeSS") {
    $key = "XESS"
  }
  if ($Name -eq "ResolutionScaling") {
    $map = @{
      "Off" = 0
      "DLSS" = 1
      "FSR2" = 2
      "FSR4" = 3
      "FSR3" = 4
      "XESS" = 5
    }
    if ($map.ContainsKey($key)) {
      return $map[$key]
    }
  }
  if ($Name -in @("DLSS", "FSR2", "FSR3", "FSR4", "XESS")) {
    $map = @{
      "Auto" = 0
      "Quality" = 1
      "Balanced" = 2
      "Performance" = 3
      "UltraPerformance" = 4
      "Ultra Performance" = 4
    }
    if ($map.ContainsKey($text)) {
      return $map[$text]
    }
  }
  if ($Name -eq "FrameGeneration") {
    $map = @{
      "Off" = 0
      "Auto" = 1
      "On" = 2
    }
    if ($map.ContainsKey($text)) {
      return $map[$text]
    }
  }
  return $null
}

function Get-CyberpunkUserSettingSnapshot([object]$Node, [string[]]$Names, [hashtable]$Out) {
  if ($null -eq $Node) {
    return
  }
  if ($Node -is [System.Array]) {
    foreach ($item in $Node) {
      Get-CyberpunkUserSettingSnapshot $item $Names $Out
    }
    return
  }
  if ($Node -is [pscustomobject]) {
    $nameProp = $Node.PSObject.Properties["name"]
    if ($null -ne $nameProp -and $Names -contains $nameProp.Value) {
      $entry = [ordered]@{}
      foreach ($propName in @("type", "value", "index", "default_value", "default_index")) {
        $prop = $Node.PSObject.Properties[$propName]
        if ($null -ne $prop) {
          $entry[$propName] = $prop.Value
        }
      }
      $Out[[string]$nameProp.Value] = $entry
    }
    foreach ($prop in @($Node.PSObject.Properties)) {
      if ($prop.Name -ne "name" -and $prop.Value -isnot [string]) {
        Get-CyberpunkUserSettingSnapshot $prop.Value $Names $Out
      }
    }
  }
}

function Set-CyberpunkUserSettingValue([object]$Node, [string]$Name, [object]$Value) {
  if ($null -eq $Node) {
    return $false
  }
  $changed = $false
  if ($Node -is [System.Array]) {
    foreach ($item in $Node) {
      if (Set-CyberpunkUserSettingValue $item $Name $Value) {
        $changed = $true
      }
    }
    return $changed
  }
  if ($Node -is [pscustomobject]) {
    $props = @($Node.PSObject.Properties)
    $nameProp = $Node.PSObject.Properties["name"]
    $valueProp = $Node.PSObject.Properties["value"]
    if ($null -ne $nameProp -and $null -ne $valueProp -and $nameProp.Value -eq $Name) {
      $Node.value = $Value
      $indexProp = $Node.PSObject.Properties["index"]
      $index = Get-CyberpunkUserSettingIndex $Name $Value
      if ($null -ne $indexProp -and $null -ne $index) {
        $Node.index = [int]$index
      }
      $changed = $true
    }
    foreach ($prop in $props) {
      if (Set-CyberpunkUserSettingValue $prop.Value $Name $Value) {
        $changed = $true
      }
    }
  }
  return $changed
}

function Apply-CyberpunkCaptureSettings([string]$SettingsPath, [string]$Scaling, [string]$Quality) {
  if ($Scaling -eq "") {
    return @{ skipped = $true }
  }
  if (-not (Test-Path -LiteralPath $SettingsPath)) {
    throw "Cyberpunk UserSettings.json not found: $SettingsPath"
  }
  $json = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
  $trackedNames = @(
    "ResolutionScaling",
    "DLSS",
    "FSR2",
    "FSR3",
    "FSR4",
    "XESS",
    "DynamicResolutionScaling",
    "FrameGeneration",
    "DLSSFrameGen",
    "FSR3_FrameGeneration",
    "XESS_FrameGeneration"
  )
  $before = @{}
  Get-CyberpunkUserSettingSnapshot $json $trackedNames $before
  $changes = @{}
  $changes["ResolutionScaling"] = Set-CyberpunkUserSettingValue $json "ResolutionScaling" $Scaling
  $changes["DynamicResolutionScaling"] = Set-CyberpunkUserSettingValue $json "DynamicResolutionScaling" $false
  $changes["FrameGeneration"] = Set-CyberpunkUserSettingValue $json "FrameGeneration" "Off"
  $changes["DLSSFrameGen"] = Set-CyberpunkUserSettingValue $json "DLSSFrameGen" $false
  $changes["FSR3_FrameGeneration"] = Set-CyberpunkUserSettingValue $json "FSR3_FrameGeneration" $false
  $changes["XESS_FrameGeneration"] = Set-CyberpunkUserSettingValue $json "XESS_FrameGeneration" $false
  if ($Scaling -eq "DLSS") {
    $changes["DLSS_BackendPreset"] = Set-CyberpunkUserSettingValue $json "DLSS_BackendPreset" "Transformer"
    $changes["DLSS"] = Set-CyberpunkUserSettingValue $json "DLSS" $Quality
  } elseif ($Scaling -eq "FSR2") {
    $changes["FSR2"] = Set-CyberpunkUserSettingValue $json "FSR2" $Quality
  } elseif ($Scaling -eq "FSR3") {
    $changes["FSR3"] = Set-CyberpunkUserSettingValue $json "FSR3" $Quality
  } elseif ($Scaling -eq "FSR4") {
    $changes["FSR4"] = Set-CyberpunkUserSettingValue $json "FSR4" $Quality
  } elseif ($Scaling -eq "XeSS") {
    $changes["XESS"] = Set-CyberpunkUserSettingValue $json "XESS" $Quality
  }
  $after = @{}
  Get-CyberpunkUserSettingSnapshot $json $trackedNames $after
  $json | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $SettingsPath -Encoding UTF8
  return @{
    skipped = $false
    scaling = $Scaling
    quality = $Quality
    changed = $changes
    before = $before
    after = $after
  }
}

function Wait-CyberpunkProcess([int[]]$BeforePids, [int]$TimeoutSeconds = 60) {
  $before = @{}
  foreach ($pid in $BeforePids) {
    $before[[int]$pid] = $true
  }
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $candidates = @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue |
      Where-Object { -not $before.ContainsKey([int]$_.Id) } |
      Sort-Object StartTime -Descending)
    if ($candidates.Count -gt 0) {
      return $candidates[0]
    }
    Start-Sleep -Milliseconds 500
  }
  return $null
}

if (-not (Test-Path -LiteralPath $CyberpunkExe)) {
  throw "Cyberpunk executable not found: $CyberpunkExe"
}

$gameDir = Split-Path -Parent $CyberpunkExe
$gameRoot = Split-Path -Parent (Split-Path -Parent $gameDir)
$loaderArtifact = if ($NoLoaderProxy) { $null } else { Join-Path $ArtifactDir $LoaderProxyName }
$ngxArtifact = Join-Path $ArtifactDir "nvngx_dlss.dll"
$slArtifact = Join-Path $ArtifactDir "sl.interposer.dll"
$fsr3Artifact = Join-Path $ArtifactDir "ffx_fsr3_x64.dll"
$fsr3UpscalerArtifact = Join-Path $ArtifactDir "ffx_fsr3upscaler_x64.dll"
$fsrBackendArtifact = Join-Path $ArtifactDir "ffx_backend_dx12_x64.dll"
$ffxApiArtifact = Join-Path $ArtifactDir "amd_fidelityfx_dx12.dll"
if ((-not $NoLoaderProxy) -and -not (Test-Path -LiteralPath $loaderArtifact)) {
  throw "Missing staged loader proxy artifact: $loaderArtifact"
}
if ((-not $NoNgxProxy) -and -not (Test-Path -LiteralPath $ngxArtifact)) {
  throw "Missing staged nvngx_dlss.dll artifact: $ngxArtifact"
}
if ($StreamlineProxy -and -not (Test-Path -LiteralPath $slArtifact)) {
  throw "Missing staged Streamline proxy artifact: $slArtifact"
}
if ($Fsr3Proxy -and -not (Test-Path -LiteralPath $fsr3Artifact)) {
  throw "Missing staged FSR3 proxy artifact: $fsr3Artifact"
}
if ($Fsr3UpscalerProxy -and -not (Test-Path -LiteralPath $fsr3UpscalerArtifact)) {
  throw "Missing staged FSR3 upscaler proxy artifact: $fsr3UpscalerArtifact"
}
if ($FsrBackendProxy -and -not (Test-Path -LiteralPath $fsrBackendArtifact)) {
  throw "Missing staged FFX DX12 backend proxy artifact: $fsrBackendArtifact"
}
if ($FfxApiProxy -and -not (Test-Path -LiteralPath $ffxApiArtifact)) {
  throw "Missing staged generic FidelityFX API proxy artifact: $ffxApiArtifact"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$loaderPath = if ($NoLoaderProxy) { $null } else { Join-Path $gameDir $LoaderProxyName }
$loaderBackup = if ($NoLoaderProxy) { $null } else { Join-Path $gameDir "$LoaderProxyName.oss-smoke-$stamp.bak" }
$ngxPath = Join-Path $gameDir "nvngx_dlss.dll"
$ngxBackup = Join-Path $gameDir "nvngx_dlss.dll.oss-backup"
$slPath = Join-Path $gameDir "sl.interposer.dll"
$slRealPath = Join-Path $gameDir "oss_sl_real.dll"
$fsr3Path = Join-Path $gameDir "ffx_fsr3_x64.dll"
$fsr3Backup = Join-Path $gameDir "ffx_fsr3_x64.dll.oss-backup"
$fsr3UpscalerPath = Join-Path $gameDir "ffx_fsr3upscaler_x64.dll"
$fsr3UpscalerBackup = Join-Path $gameDir "ffx_fsr3upscaler_x64.dll.oss-backup"
$fsrBackendPath = Join-Path $gameDir "ffx_backend_dx12_x64.dll"
$fsrBackendBackup = Join-Path $gameDir "ffx_backend_dx12_x64.dll.oss-backup"
$ffxApiPath = Join-Path $gameDir "amd_fidelityfx_dx12.DLL"
$ffxApiBackup = Join-Path $gameDir "amd_fidelityfx_dx12.DLL.oss-backup"
$createdNgxBackup = $false
$createdSlBackup = $false
$createdFsr3Backup = $false
$createdFsr3UpscalerBackup = $false
$createdFsrBackendBackup = $false
$createdFfxApiBackup = $false
$hadLoader = if ($NoLoaderProxy) { $false } else { Test-Path -LiteralPath $loaderPath }
$hadNgx = if ($NoNgxProxy) { $false } else { Test-Path -LiteralPath $ngxPath }
$hadSl = Test-Path -LiteralPath $slPath
$hadSlReal = Test-Path -LiteralPath $slRealPath
$hadFsr3 = if ($Fsr3Proxy) { Test-Path -LiteralPath $fsr3Path } else { $false }
$hadFsr3Upscaler = if ($Fsr3UpscalerProxy) { Test-Path -LiteralPath $fsr3UpscalerPath } else { $false }
$hadFsrBackend = if ($FsrBackendProxy) { Test-Path -LiteralPath $fsrBackendPath } else { $false }
$hadFfxApi = if ($FfxApiProxy) { Test-Path -LiteralPath $ffxApiPath } else { $false }
$clearedNgxBackup = if ($NoNgxProxy) { $false } else { Clear-Or-Refuse-StaleBackup $ngxPath $ngxBackup }
$clearedSlReal = if ($StreamlineProxy) { Clear-Or-Refuse-StaleBackup $slPath $slRealPath } else { $false }
$clearedFsr3Backup = if ($Fsr3Proxy) { Clear-Or-Refuse-StaleBackup $fsr3Path $fsr3Backup } else { $false }
$clearedFsr3UpscalerBackup = if ($Fsr3UpscalerProxy) { Clear-Or-Refuse-StaleBackup $fsr3UpscalerPath $fsr3UpscalerBackup } else { $false }
$clearedFsrBackendBackup = if ($FsrBackendProxy) { Clear-Or-Refuse-StaleBackup $fsrBackendPath $fsrBackendBackup } else { $false }
$clearedFfxApiBackup = if ($FfxApiProxy) { Clear-Or-Refuse-StaleBackup $ffxApiPath $ffxApiBackup } else { $false }
$hadNgxBackup = if ($NoNgxProxy) { $false } else { Test-Path -LiteralPath $ngxBackup }
$hadSlReal = Test-Path -LiteralPath $slRealPath
$hadFsr3Backup = if ($Fsr3Proxy) { Test-Path -LiteralPath $fsr3Backup } else { $false }
$hadFsr3UpscalerBackup = if ($Fsr3UpscalerProxy) { Test-Path -LiteralPath $fsr3UpscalerBackup } else { $false }
$hadFsrBackendBackup = if ($FsrBackendProxy) { Test-Path -LiteralPath $fsrBackendBackup } else { $false }
$hadFfxApiBackup = if ($FfxApiProxy) { Test-Path -LiteralPath $ffxApiBackup } else { $false }
$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$configDir = Join-Path $localAppData "oss-capture"
$pendingRoot = "E:\OSS-Capture\pending"
$configPath = Join-Path $configDir "config.json"
$configBackup = Join-Path $configDir "config.json.oss-smoke-$stamp.bak"
$userSettingsPath = Join-Path $localAppData "CD Projekt Red\Cyberpunk 2077\UserSettings.json"
$userSettingsBackup = Join-Path $localAppData "CD Projekt Red\Cyberpunk 2077\UserSettings.json.oss-smoke-$stamp.bak"
$logPath = Join-Path (Join-Path $localAppData "oss-gaussian") "interception.log"
$pendingBefore = 0
if (Test-Path -LiteralPath $pendingRoot) {
  $pendingBefore = @(Get-ChildItem -LiteralPath $pendingRoot -Recurse -File -ErrorAction SilentlyContinue).Count
}

$proc = $null
$launchError = $null
$modBackups = @()
$oldInitMarker = [Environment]::GetEnvironmentVariable("OSS_GAUSSIAN_INIT_MARKER", "Process")
$oldInitInDllMain = [Environment]::GetEnvironmentVariable("OSS_GAUSSIAN_INIT_IN_DLLMAIN", "Process")
$markerPath = Join-Path (Join-Path $localAppData "oss-gaussian") "init-marker.log"
$result = [ordered]@{
  status = "unknown"
  cyberpunk_exe = $CyberpunkExe
  artifact_dir = $ArtifactDir
  loader_proxy_name = $LoaderProxyName
  no_loader_proxy = [bool]$NoLoaderProxy
  no_ngx_proxy = [bool]$NoNgxProxy
  streamline_proxy = [bool]$StreamlineProxy
  fsr3_proxy = [bool]$Fsr3Proxy
  fsr3_upscaler_proxy = [bool]$Fsr3UpscalerProxy
  fsr_backend_proxy = [bool]$FsrBackendProxy
  ffx_api_proxy = [bool]$FfxApiProxy
  temporarily_disable_mods = [bool]$TemporarilyDisableMods
  launch_via_steam = [bool]$LaunchViaSteam
  steam_exe = $SteamExe
  steam_app_id = $SteamAppId
  init_marker_enabled = [bool]$EnableInitMarker
  init_in_dllmain = [bool]$InitInDllMain
  game_dir = $gameDir
  game_root = $gameRoot
  launch_args = $LaunchArgs
  waited_seconds = $WaitSeconds
  loader_before = if ($NoLoaderProxy) { @{ skipped = $true } } else { Get-FileState $loaderPath }
  nvngx_before = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxPath }
  nvngx_backup_before = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxBackup }
  streamline_before = Get-FileState $slPath
  streamline_real_before = Get-FileState $slRealPath
  fsr3_before = if ($Fsr3Proxy) { Get-FileState $fsr3Path } else { @{ skipped = $true } }
  fsr3_backup_before = if ($Fsr3Proxy) { Get-FileState $fsr3Backup } else { @{ skipped = $true } }
  fsr3_upscaler_before = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerPath } else { @{ skipped = $true } }
  fsr3_upscaler_backup_before = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerBackup } else { @{ skipped = $true } }
  fsr_backend_before = if ($FsrBackendProxy) { Get-FileState $fsrBackendPath } else { @{ skipped = $true } }
  fsr_backend_backup_before = if ($FsrBackendProxy) { Get-FileState $fsrBackendBackup } else { @{ skipped = $true } }
  ffx_api_before = if ($FfxApiProxy) { Get-FileState $ffxApiPath } else { @{ skipped = $true } }
  ffx_api_backup_before = if ($FfxApiProxy) { Get-FileState $ffxApiBackup } else { @{ skipped = $true } }
  user_settings_before = Get-FileState $userSettingsPath
  force_resolution_scaling = $ForceResolutionScaling
  force_upscaler_quality = $ForceUpscalerQuality
  cleared_stale_backups = @{
    nvngx = [bool]$clearedNgxBackup
    streamline_real = [bool]$clearedSlReal
    fsr3 = [bool]$clearedFsr3Backup
    fsr3_upscaler = [bool]$clearedFsr3UpscalerBackup
    fsr_backend = [bool]$clearedFsrBackendBackup
    ffx_api = [bool]$clearedFfxApiBackup
  }
}

try {
  Write-Trace "prekill-start"
  Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  if (-not (Wait-NoCyberpunkProcess 30)) {
    throw "Cyberpunk2077.exe is still running; refusing to mutate game DLLs"
  }
  Write-Trace "prekill-done"

  New-Item -ItemType Directory -Force -Path $configDir | Out-Null
  if (Test-Path -LiteralPath $configPath) {
    Copy-Item -LiteralPath $configPath -Destination $configBackup -Force
  }
  if ($ForceResolutionScaling -ne "") {
    if (-not (Test-Path -LiteralPath $userSettingsPath)) {
      throw "Cyberpunk UserSettings.json not found: $userSettingsPath"
    }
    Copy-Item -LiteralPath $userSettingsPath -Destination $userSettingsBackup -Force
    $result.user_settings_override = Apply-CyberpunkCaptureSettings $userSettingsPath $ForceResolutionScaling $ForceUpscalerQuality
  } else {
    $result.user_settings_override = @{ skipped = $true }
  }
  if (Test-Path -LiteralPath $logPath) {
    Remove-Item -LiteralPath $logPath -Force
  }
  if (Test-Path -LiteralPath $markerPath) {
    Remove-Item -LiteralPath $markerPath -Force
  }

  if ($TemporarilyDisableMods) {
    Write-Trace "mod-disable-start"
    $modTargets = @(
      (Join-Path $gameRoot "red4ext\RED4ext.dll"),
      (Join-Path $gameDir "winmm.dll"),
      (Join-Path $gameDir "version.dll")
    )
    if ($NoLoaderProxy) {
      $modTargets += (Join-Path $gameDir "dxgi.dll")
    }
    foreach ($modPath in $modTargets) {
      $disabled = Disable-ModFileForSmoke $modPath $stamp
      if ($null -ne $disabled) {
        $modBackups += $disabled
      }
    }
    $result.mod_backups = $modBackups
    Write-Trace "mod-disable-done"
  }

  Write-Trace "backup-start"
  if ((-not $NoLoaderProxy) -and $hadLoader) {
    Copy-Item -LiteralPath $loaderPath -Destination $loaderBackup -Force
  }
  if ((-not $NoNgxProxy) -and $hadNgx -and -not $hadNgxBackup) {
    Copy-Item -LiteralPath $ngxPath -Destination $ngxBackup -Force
    $createdNgxBackup = $true
  }
  if ($StreamlineProxy -and $hadSl -and -not $hadSlReal) {
    Copy-Item -LiteralPath $slPath -Destination $slRealPath -Force
    $createdSlBackup = $true
  }
  if ($Fsr3Proxy -and $hadFsr3 -and -not $hadFsr3Backup) {
    Copy-Item -LiteralPath $fsr3Path -Destination $fsr3Backup -Force
    $createdFsr3Backup = $true
  }
  if ($Fsr3UpscalerProxy -and $hadFsr3Upscaler -and -not $hadFsr3UpscalerBackup) {
    Copy-Item -LiteralPath $fsr3UpscalerPath -Destination $fsr3UpscalerBackup -Force
    $createdFsr3UpscalerBackup = $true
  }
  if ($FsrBackendProxy -and $hadFsrBackend -and -not $hadFsrBackendBackup) {
    Copy-Item -LiteralPath $fsrBackendPath -Destination $fsrBackendBackup -Force
    $createdFsrBackendBackup = $true
  }
  if ($FfxApiProxy -and $hadFfxApi -and -not $hadFfxApiBackup) {
    Copy-Item -LiteralPath $ffxApiPath -Destination $ffxApiBackup -Force
    $createdFfxApiBackup = $true
  }
  Write-Trace "backup-done"

  Write-Trace "install-start"
  if (-not $NoLoaderProxy) {
    Copy-Item -LiteralPath $loaderArtifact -Destination $loaderPath -Force
  }
  if (-not $NoNgxProxy) {
    Copy-Item -LiteralPath $ngxArtifact -Destination $ngxPath -Force
  }
  if ($StreamlineProxy) {
    Copy-Item -LiteralPath $slArtifact -Destination $slPath -Force
  }
  if ($Fsr3Proxy) {
    Copy-Item -LiteralPath $fsr3Artifact -Destination $fsr3Path -Force
  }
  if ($Fsr3UpscalerProxy) {
    Copy-Item -LiteralPath $fsr3UpscalerArtifact -Destination $fsr3UpscalerPath -Force
  }
  if ($FsrBackendProxy) {
    Copy-Item -LiteralPath $fsrBackendArtifact -Destination $fsrBackendPath -Force
  }
  if ($FfxApiProxy) {
    Copy-Item -LiteralPath $ffxApiArtifact -Destination $ffxApiPath -Force
  }
  $result.loader_installed = if ($NoLoaderProxy) { @{ skipped = $true } } else { Get-FileState $loaderPath }
  $result.nvngx_installed = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxPath }
  $result.streamline_installed = if ($StreamlineProxy) { Get-FileState $slPath } else { @{ skipped = $true } }
  $result.streamline_real_installed = Get-FileState $slRealPath
  $result.fsr3_installed = if ($Fsr3Proxy) { Get-FileState $fsr3Path } else { @{ skipped = $true } }
  $result.fsr3_backup_installed = if ($Fsr3Proxy) { Get-FileState $fsr3Backup } else { @{ skipped = $true } }
  $result.fsr3_upscaler_installed = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerPath } else { @{ skipped = $true } }
  $result.fsr3_upscaler_backup_installed = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerBackup } else { @{ skipped = $true } }
  $result.fsr_backend_installed = if ($FsrBackendProxy) { Get-FileState $fsrBackendPath } else { @{ skipped = $true } }
  $result.fsr_backend_backup_installed = if ($FsrBackendProxy) { Get-FileState $fsrBackendBackup } else { @{ skipped = $true } }
  $result.ffx_api_installed = if ($FfxApiProxy) { Get-FileState $ffxApiPath } else { @{ skipped = $true } }
  $result.ffx_api_backup_installed = if ($FfxApiProxy) { Get-FileState $ffxApiBackup } else { @{ skipped = $true } }
  Write-Trace "install-done"

  $config = [ordered]@{
    game_id = "cyberpunk-2077"
    game_exe_name = "Cyberpunk2077.exe"
    proxy_dll_name = $LoaderProxyName
    capture_mode = $CaptureMode
    capture_storage_mode = "local"
    pending_dir = "E:\OSS-Capture\pending"
    output_dir = "E:\OSS-Capture"
    capture_api_base = "http://127.0.0.1:8080"
    install_token = "local-cyberpunk-smoke-$stamp"
    enabled = $true
  }
  $config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $configDir "config.json") -Encoding UTF8

  if ($EnableInitMarker) {
    [Environment]::SetEnvironmentVariable("OSS_GAUSSIAN_INIT_MARKER", "1", "Process")
  }
  if ($InitInDllMain) {
    [Environment]::SetEnvironmentVariable("OSS_GAUSSIAN_INIT_IN_DLLMAIN", "1", "Process")
  }

  Write-Trace "launch-start"
  $preLaunchPids = @((Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue | ForEach-Object { [int]$_.Id }))
  try {
    if ($LaunchViaSteam) {
      if (-not (Test-Path -LiteralPath $SteamExe)) {
        throw "Steam executable not found: $SteamExe"
      }
      $steamArgs = @("-applaunch", $SteamAppId) + $LaunchArgs
      $steamProc = Start-Process -FilePath $SteamExe -ArgumentList $steamArgs -PassThru
      Write-Trace "steam-launch pid=$($steamProc.Id) args=$($steamArgs -join ' ')"
      $proc = Wait-CyberpunkProcess $preLaunchPids 90
      if ($null -eq $proc) {
        throw "Steam launch did not create Cyberpunk2077.exe within 90 seconds"
      }
    } else {
      $proc = Start-Process -FilePath $CyberpunkExe -ArgumentList $LaunchArgs -WorkingDirectory $gameDir -PassThru
    }
  } catch {
    $launchError = $_.Exception.Message
  }
  Write-Trace "launch-done pid=$($proc.Id) error=$launchError"

  if ($proc -ne $null) {
    Write-Trace "wait-start seconds=$WaitSeconds"
    Start-Sleep -Seconds $WaitSeconds
    Write-Trace "wait-done"
  }

  Write-Trace "inspect-start"
  $running = @(Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue)
  $launchedStillRunning = $false
  if ($proc -ne $null) {
    $launchedStillRunning = [bool](Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)
  }
  $loadedModules = if ($launchedStillRunning) { Get-CyberpunkLoadedModules $proc.Id } else { @() }
  $loadedModules = @($loadedModules | ForEach-Object {
    [ordered]@{
      process_id = [int]$_.process_id
      module_name = [string]$_.module_name
      file_name = [string]$_.file_name
    }
  })
  Write-Trace "inspect-modules-done"
  $result.loader_during_launch = if ($NoLoaderProxy) { @{ skipped = $true } } else { Get-FileState $loaderPath }
  $result.nvngx_during_launch = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxPath }
  $result.streamline_during_launch = Get-FileState $slPath
  $result.streamline_real_during_launch = Get-FileState $slRealPath
  $result.fsr3_during_launch = if ($Fsr3Proxy) { Get-FileState $fsr3Path } else { @{ skipped = $true } }
  $result.fsr3_backup_during_launch = if ($Fsr3Proxy) { Get-FileState $fsr3Backup } else { @{ skipped = $true } }
  $result.fsr3_upscaler_during_launch = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerPath } else { @{ skipped = $true } }
  $result.fsr3_upscaler_backup_during_launch = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerBackup } else { @{ skipped = $true } }
  $result.fsr_backend_during_launch = if ($FsrBackendProxy) { Get-FileState $fsrBackendPath } else { @{ skipped = $true } }
  $result.fsr_backend_backup_during_launch = if ($FsrBackendProxy) { Get-FileState $fsrBackendBackup } else { @{ skipped = $true } }
  $result.ffx_api_during_launch = if ($FfxApiProxy) { Get-FileState $ffxApiPath } else { @{ skipped = $true } }
  $result.ffx_api_backup_during_launch = if ($FfxApiProxy) { Get-FileState $ffxApiBackup } else { @{ skipped = $true } }
  Write-Trace "inspect-files-done"
  $pendingAfter = 0
  $pendingSamples = @()
  if (Test-Path -LiteralPath $pendingRoot) {
    $files = @(Get-ChildItem -LiteralPath $pendingRoot -Recurse -File -ErrorAction SilentlyContinue)
    $pendingAfter = $files.Count
    $pendingSamples = @($files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 12 | ForEach-Object {
      $_.FullName
    })
  }

  $logTail = Read-LogTail $logPath
  $markerTail = Read-LogTail $markerPath
  Write-Trace "inspect-logs-done"
  $captureFilesDelta = $pendingAfter - $pendingBefore
  $captureValidation = if ($captureFilesDelta -gt 0) {
    Write-Trace "capture-validation-start"
    $validation = Invoke-CaptureValidation $pendingRoot
    Write-Trace "capture-validation-done valid=$($validation.valid) skipped=$($validation.skipped)"
    $validation
  } else {
    @{ skipped = $true; reason = "no new pending files"; pending_delta = $captureFilesDelta }
  }
  $hookLoaded = (($logTail + $markerTail) -join "`n") -match "OSS Gaussian|oss-capture|NGX|dxgi|dllmain-attach|begin|end|ffx|FSR3"
  $result.status = if ($launchError -ne $null) {
    "launch_error"
  } elseif ($captureFilesDelta -gt 0 -and
            $captureValidation.ContainsKey("valid") -and
            $captureValidation.valid -eq $false) {
    "captured_invalid"
  } elseif ($captureFilesDelta -gt 0) {
    "captured"
  } elseif ($hookLoaded) {
    "hook_loaded_no_capture"
  } elseif ($proc -ne $null) {
    "launched_no_hook_evidence"
  } else {
    "not_launched"
  }
  $result.process_id = if ($proc -ne $null) { $proc.Id } else { $null }
  $result.launch_error = $launchError
  $result.process_running_count = $running.Count
  $result.loaded_modules = $loadedModules
  $result.pending_before = $pendingBefore
  $result.pending_after = $pendingAfter
  $result.pending_delta = $captureFilesDelta
  $result.pending_samples = $pendingSamples
  $result.capture_validation = $captureValidation
  $result.log_path = $logPath
  $result.log_tail = $logTail
  $result.init_marker_path = $markerPath
  $result.init_marker_tail = $markerTail
} finally {
  Write-Trace "finally-start"
  [Environment]::SetEnvironmentVariable("OSS_GAUSSIAN_INIT_MARKER", $oldInitMarker, "Process")
  [Environment]::SetEnvironmentVariable("OSS_GAUSSIAN_INIT_IN_DLLMAIN", $oldInitInDllMain, "Process")
  Get-Process -Name "Cyberpunk2077" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Wait-NoCyberpunkProcess 30 | Out-Null
  Write-Trace "process-stop-done"

  if ((-not $NoLoaderProxy) -and $hadLoader) {
    Copy-Item -LiteralPath $loaderBackup -Destination $loaderPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $loaderBackup -Force -ErrorAction SilentlyContinue
  } elseif (-not $NoLoaderProxy) {
    Remove-Item -LiteralPath $loaderPath -Force -ErrorAction SilentlyContinue
  }

  if ((-not $NoNgxProxy) -and $hadNgx) {
    if (Test-Path -LiteralPath $ngxBackup) {
      Copy-Item -LiteralPath $ngxBackup -Destination $ngxPath -Force -ErrorAction SilentlyContinue
    }
    if ($createdNgxBackup) {
      Remove-Item -LiteralPath $ngxBackup -Force -ErrorAction SilentlyContinue
    }
  } elseif (-not $NoNgxProxy) {
    Remove-Item -LiteralPath $ngxPath -Force -ErrorAction SilentlyContinue
  }

  if ($StreamlineProxy) {
    if ($hadSl -and (Test-Path -LiteralPath $slRealPath)) {
      Copy-Item -LiteralPath $slRealPath -Destination $slPath -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadSl) {
      Remove-Item -LiteralPath $slPath -Force -ErrorAction SilentlyContinue
    }
    if ($createdSlBackup) {
      Remove-Item -LiteralPath $slRealPath -Force -ErrorAction SilentlyContinue
    }
  }

  if ($Fsr3Proxy) {
    if ($hadFsr3 -and (Test-Path -LiteralPath $fsr3Backup)) {
      Copy-Item -LiteralPath $fsr3Backup -Destination $fsr3Path -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadFsr3) {
      Remove-Item -LiteralPath $fsr3Path -Force -ErrorAction SilentlyContinue
    }
    if ($createdFsr3Backup) {
      Remove-Item -LiteralPath $fsr3Backup -Force -ErrorAction SilentlyContinue
    }
  }

  if ($Fsr3UpscalerProxy) {
    if ($hadFsr3Upscaler -and (Test-Path -LiteralPath $fsr3UpscalerBackup)) {
      Copy-Item -LiteralPath $fsr3UpscalerBackup -Destination $fsr3UpscalerPath -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadFsr3Upscaler) {
      Remove-Item -LiteralPath $fsr3UpscalerPath -Force -ErrorAction SilentlyContinue
    }
    if ($createdFsr3UpscalerBackup) {
      Remove-Item -LiteralPath $fsr3UpscalerBackup -Force -ErrorAction SilentlyContinue
    }
  }

  if ($FsrBackendProxy) {
    if ($hadFsrBackend -and (Test-Path -LiteralPath $fsrBackendBackup)) {
      Copy-Item -LiteralPath $fsrBackendBackup -Destination $fsrBackendPath -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadFsrBackend) {
      Remove-Item -LiteralPath $fsrBackendPath -Force -ErrorAction SilentlyContinue
    }
    if ($createdFsrBackendBackup) {
      Remove-Item -LiteralPath $fsrBackendBackup -Force -ErrorAction SilentlyContinue
    }
  }

  if ($FfxApiProxy) {
    if ($hadFfxApi -and (Test-Path -LiteralPath $ffxApiBackup)) {
      Copy-Item -LiteralPath $ffxApiBackup -Destination $ffxApiPath -Force -ErrorAction SilentlyContinue
    } elseif (-not $hadFfxApi) {
      Remove-Item -LiteralPath $ffxApiPath -Force -ErrorAction SilentlyContinue
    }
    if ($createdFfxApiBackup) {
      Remove-Item -LiteralPath $ffxApiBackup -Force -ErrorAction SilentlyContinue
    }
  }

  foreach ($mod in @($modBackups)) {
    try {
      if (Test-Path -LiteralPath $mod.backup) {
        Move-Item -LiteralPath $mod.backup -Destination $mod.path -Force
      }
    } catch {
      $result.restore_error = "mod restore failed: $($_.Exception.Message)"
    }
  }

  if (Test-Path -LiteralPath $configBackup) {
    Copy-Item -LiteralPath $configBackup -Destination $configPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $configBackup -Force -ErrorAction SilentlyContinue
  } elseif (Test-Path -LiteralPath $configPath) {
    Remove-Item -LiteralPath $configPath -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path -LiteralPath $userSettingsBackup) {
    Copy-Item -LiteralPath $userSettingsBackup -Destination $userSettingsPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $userSettingsBackup -Force -ErrorAction SilentlyContinue
  }
  Write-Trace "restore-done"

  $result.loader_after_restore = if ($NoLoaderProxy) { @{ skipped = $true } } else { Get-FileState $loaderPath }
  $result.nvngx_after_restore = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxPath }
  $result.nvngx_backup_after_restore = if ($NoNgxProxy) { @{ skipped = $true } } else { Get-FileState $ngxBackup }
  $result.streamline_after_restore = Get-FileState $slPath
  $result.streamline_real_after_restore = Get-FileState $slRealPath
  $result.fsr3_after_restore = if ($Fsr3Proxy) { Get-FileState $fsr3Path } else { @{ skipped = $true } }
  $result.fsr3_backup_after_restore = if ($Fsr3Proxy) { Get-FileState $fsr3Backup } else { @{ skipped = $true } }
  $result.fsr3_upscaler_after_restore = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerPath } else { @{ skipped = $true } }
  $result.fsr3_upscaler_backup_after_restore = if ($Fsr3UpscalerProxy) { Get-FileState $fsr3UpscalerBackup } else { @{ skipped = $true } }
  $result.fsr_backend_after_restore = if ($FsrBackendProxy) { Get-FileState $fsrBackendPath } else { @{ skipped = $true } }
  $result.fsr_backend_backup_after_restore = if ($FsrBackendProxy) { Get-FileState $fsrBackendBackup } else { @{ skipped = $true } }
  $result.ffx_api_after_restore = if ($FfxApiProxy) { Get-FileState $ffxApiPath } else { @{ skipped = $true } }
  $result.ffx_api_backup_after_restore = if ($FfxApiProxy) { Get-FileState $ffxApiBackup } else { @{ skipped = $true } }
  $result.user_settings_after_restore = Get-FileState $userSettingsPath
  Write-Trace "finally-done"
}

Write-Trace "json-start"
$json = ConvertTo-SmokeJson $result
if ($ReportPath -ne "") {
  $reportDir = Split-Path -Parent $ReportPath
  if ($reportDir -ne "") {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
  }
  $json | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}
Write-Trace "json-done"
$json

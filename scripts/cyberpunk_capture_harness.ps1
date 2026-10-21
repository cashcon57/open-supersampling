param(
  [string]$HostName = "",
  [string]$RepoPath = "C:\Users\$env:USERNAME\OpenSuperSampling",
  [string]$Python = "python",
  [string]$Server = "http://127.0.0.1:8080",
  [string]$Token = "",
  [string]$OutputDir = "",
  [string]$CyberpunkExe = "",
  [int]$Frames = 1,
  [switch]$StartIngest,
  [switch]$KeepIngestRunning,
  [switch]$MintLocalToken,
  [switch]$LaunchCyberpunk,
  [switch]$SmokeFakeCyberpunkInstall,
  [switch]$KeepSmokeWorkdir
)

$ErrorActionPreference = "Stop"

function New-HarnessArgs {
  $args = @(
    "scripts/capture_cyberpunk_harness.py",
    "--server", $Server,
    "--frames", [string]$Frames
  )
  if ($Token -ne "") {
    $args += @("--token", $Token)
  }
  if ($OutputDir -ne "") {
    $args += @("--output-dir", $OutputDir)
  }
  if ($CyberpunkExe -ne "") {
    $args += @("--cyberpunk-exe", $CyberpunkExe)
  }
  if ($StartIngest) {
    $args += "--start-ingest"
  }
  if ($KeepIngestRunning) {
    $args += "--keep-ingest-running"
  }
  if ($MintLocalToken) {
    $args += "--mint-local-token"
  }
  if ($LaunchCyberpunk) {
    $args += "--launch-cyberpunk"
  }
  if ($SmokeFakeCyberpunkInstall) {
    $args += "--smoke-fake-cyberpunk-install"
  }
  if ($KeepSmokeWorkdir) {
    $args += "--keep-smoke-workdir"
  }
  return $args
}

$HarnessArgs = New-HarnessArgs

if ($HostName -eq "") {
  Set-Location $RepoPath
  & $Python @HarnessArgs
  exit $LASTEXITCODE
}

Invoke-Command -HostName $HostName -ArgumentList $RepoPath, $Python, $HarnessArgs -ScriptBlock {
  param($RemoteRepoPath, $RemotePython, $RemoteHarnessArgs)
  Set-Location $RemoteRepoPath
  & $RemotePython @RemoteHarnessArgs
  exit $LASTEXITCODE
}

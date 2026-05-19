$ErrorActionPreference = "Continue"

$repo = "E:\oss-gaussian"
$python = "C:\Users\cashc\Miniconda3\envs\image-gs\python.exe"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True removed — Windows
# torch wheels don't support it; setting it kills the process silently
# after the "not supported on this platform" warning.

# vcvarsall.bat sourcing for torch.compile cl.exe SKIPPED — see launch_v7_debug.ps1.

# Scan-and-kill any surviving sr_train_v7 python processes by command line.
# Replaces the previous hardcoded $pid_to_kill which went stale on every run.
$existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -like '*sr_train_v7*' }
if ($existing) {
  foreach ($p in $existing) {
    Write-Output ("Stopping sr_train_v7 PID " + $p.ProcessId)
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
    } catch {
      Write-Output ("  stop failed: " + $_.Exception.Message)
    }
  }
  Start-Sleep -Seconds 3
} else {
  Write-Output "No sr_train_v7 processes running."
}

$argList = @(
  "--tartanair-root", "E:/datasets/tartanair_extracted",
  "--output-dir", "E:\checkpoints\srcnn-v7.0-pico-005",
  "--steps", "100000",
  "--batch-size", "2",
  "--device", "cuda",
  "--log-every", "50",
  "--ckpt-every", "500",
  "--ckpt-warmup-steps", "100,500,1000,2000",
  "--backbone-kind", "hat_tiny",
  "--curriculum",
  "--enable-parent-child",
  "--parent-child-drift-rate", "0.05",
  "--canvas-capacity", "16384",
  "--max-hr-crop", "384",
  "--no-compile"
) -join " "

$stdoutLog = Join-Path $logDir "v7-pico-005-restart-batch2-crop384.log"
$stderrLog = Join-Path $logDir "v7-pico-005-restart-batch2-crop384.err.log"

# Start-Process inherits the launching shell's environment (PATH includes
# the conda env's Library\bin so cudnn DLLs load). WMI Win32_Process.Create
# does NOT inherit user env and fails silently before main() runs.
Write-Output ("Launching: " + $python + " scripts\sr_train_v7.py " + $argList)
# Use Start-Process with -RedirectStandardOutput/Error (worked on torch 2.4.1).
$proc = Start-Process -FilePath $python `
  -ArgumentList "-u -X faulthandler scripts\sr_train_v7.py $argList" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -WindowStyle Hidden `
  -PassThru

Write-Output ("Launched PID: " + $proc.Id)
Start-Sleep -Seconds 10
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
  Write-Output "Still alive after 10s."
  Write-Output ("Stdout log: " + $stdoutLog)
  Write-Output ("Stderr log: " + $stderrLog)
} else {
  Write-Output "DIED within 10s. Stderr tail:"
  Get-Content $stderrLog -Tail 30 -ErrorAction SilentlyContinue
  Write-Output "Stdout tail:"
  Get-Content $stdoutLog -Tail 30 -ErrorAction SilentlyContinue
  exit 1
}

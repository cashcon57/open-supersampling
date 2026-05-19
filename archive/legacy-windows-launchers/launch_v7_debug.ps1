$ErrorActionPreference = "Continue"
$repo = "E:\oss-gaussian"
$python = "C:\Users\cashc\Miniconda3\envs\image-gs\python.exe"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was tried here but
# Windows torch wheels don't support expandable_segments — setting it
# emits a "not supported on this platform" warning and then causes the
# CUDA allocator to abort the process silently (zero stdout, only the
# warning in stderr). Removed. Use Linux for expandable_segments.

# vcvarsall.bat sourcing for torch.compile cl.exe SKIPPED — it produced
# silent process deaths (env table appears to break Start-Process child
# stdout). When --no-compile is set on the trainer we don't need MSVC
# anyway. Re-enable this block only when compile is on AND we've found
# a more reliable env-passing mechanism (e.g. cmd /c launch wrapper).
$logDir = "E:\oss-gaussian\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = "$logDir\v7-debug.log"

# First, quick smoke test: just run --help to confirm import works.
Push-Location $repo
$smoke = & $python "scripts\sr_train_v7.py" --help 2>&1
Pop-Location
Write-Output ("--help exit code: " + $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
  Write-Output "--- --help OUTPUT (first 60 lines) ---"
  $smoke | Select-Object -First 60
  exit 1
}
Write-Output "--help OK. Script imports cleanly."

# Now do the real WMI launch
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

# Use Start-Process with -RedirectStandardOutput/Error. This worked on torch
# 2.4.1 earlier today (PID 26124 / 3292 both reached step 550 cleanly). The
# cmd /c wrapper variant we briefly tried produced silent child deaths with
# empty logs — appears to be a quote-parsing issue in cmd's interaction with
# Start-Process's ArgumentList array form.
$proc = Start-Process -FilePath $python -ArgumentList "-u -X faulthandler scripts\sr_train_v7.py $argList" `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $logPath `
  -RedirectStandardError "$logDir\v7-debug.err.log" `
  -WindowStyle Hidden `
  -PassThru
Write-Output ("Launched PID: " + $proc.Id)
Start-Sleep -Seconds 10
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
  Write-Output "Still alive after 10s"
} else {
  Write-Output "DIED within 10s. Stderr tail:"
  Get-Content "$logDir\v7-debug.err.log" -Tail 30 -ErrorAction SilentlyContinue
  Write-Output "Stdout tail:"
  Get-Content $logPath -Tail 30 -ErrorAction SilentlyContinue
}

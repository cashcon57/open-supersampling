$p = Get-Process -Id 14772 -ErrorAction SilentlyContinue
if ($p) {
  Write-Output ("PID 14772 alive. CPU=" + $p.CPU + " WS=" + [math]::Round($p.WorkingSet64/1MB,1) + "MB")
} else {
  Write-Output "PID 14772 NOT running"
}

# Check for any sr_train_v7 children (cmd.exe spawns python.exe; the original PID was cmd.exe)
$py = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -like '*sr_train_v7*' }
foreach ($x in $py) {
  Write-Output ("python PID=" + $x.ProcessId + "  ParentPID=" + $x.ParentProcessId)
}

# Check log file existence
$log = "E:\oss-gaussian\logs\v7-pico-005-restart-batch2-crop384.log"
if (Test-Path $log) {
  Write-Output ("LOG: " + (Get-Item $log).Length + " bytes")
  Write-Output "--- TAIL ---"
  Get-Content $log -Tail 30
} else {
  Write-Output "LOG MISSING: $log"
  Write-Output "Directory contents:"
  Get-ChildItem "E:\oss-gaussian\logs" -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime
}

# Check VRAM
Write-Output "--- GPU ---"
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1

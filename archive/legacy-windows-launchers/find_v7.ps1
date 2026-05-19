$procs = Get-CimInstance Win32_Process -Filter "name = 'python.exe'"
foreach ($p in $procs) {
  if ($p.CommandLine -like '*sr_train_v7*') {
    Write-Output ("PID=" + $p.ProcessId)
    Write-Output ("CMD=" + $p.CommandLine)
    Write-Output "---"
  }
}

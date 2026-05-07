$cmd = 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\cashc\viz-daemon-supervisor.ps1'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
$r | Select ProcessId, ReturnValue

$cmd = 'powershell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File C:\Users\cashc\viz-daemon-supervisor.ps1'
$startupHidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; ProcessStartupInformation = $startupHidden }
$r | Select ProcessId, ReturnValue

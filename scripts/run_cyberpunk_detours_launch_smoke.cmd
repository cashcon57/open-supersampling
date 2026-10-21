@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\Temp\cyberpunk_detours_launch_smoke.ps1 -WaitSeconds 60 -ReportPath E:\Temp\cyberpunk_detours_launch_smoke_report.json > E:\Temp\cyberpunk_detours_launch_smoke_stdout.txt 2>&1

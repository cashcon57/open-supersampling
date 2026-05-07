# 3080ti host scripts (PowerShell)

These scripts live on the Windows training host at `C:\Users\cashc\` and are
mirrored here for version control. They wrap the inflight viz renderer with
WMI orphan-spawn so the processes survive SSH disconnect.

## viz daemon

`start-viz-daemon.ps1` — spawns `sr_temporal_inflight_viz.py` in a polling
loop (`--interval 60`) that auto-renders any new ckpt that lands in
`E:\checkpoints\srcnn-v6.1-pico-001\`. CPU mode so it doesn't contend with
training GPU.

`viz-daemon-supervisor.ps1` — checks every 60s whether the daemon is alive;
respawns it if missing. Logs to `E:\logs\viz-supervisor.log`. Run as a
detached background process so it survives operator logout.

`start-supervisor.ps1` — convenience wrapper to orphan-spawn the supervisor
itself via `Invoke-CimMethod Win32_Process Create`.

## one-shot render

`render-step.ps1 -ckpt <full-path-to-step-NNNNNNNN.pt>` — renders viz for a
specific checkpoint. Use to backfill historical checkpoints that the daemon
missed.

## startup

Run after Windows boot OR when the supervisor process dies:

```powershell
# Start supervisor (which will start the daemon)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\cashc\start-supervisor.ps1
```

The supervisor itself currently has no auto-restart; that's a Task Scheduler
job to add later. For now it persists as long as the host doesn't reboot.

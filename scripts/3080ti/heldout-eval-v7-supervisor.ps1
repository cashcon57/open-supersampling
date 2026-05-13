# Held-out evaluation supervisor (v7).
#
# Runs sr_eval_v7.py on every NEW v7 checkpoint that lands in the active
# v7 run's directory and appends the v7 eval JSON row to score_log_v7.json
# (which the v7 dashboard lane polls). Idempotent: skips checkpoints whose
# step is already present in score_log_v7.json.
#
# Runs IN PARALLEL with heldout-eval-supervisor.ps1 (the v6 supervisor):
# the two supervisors target different runs, use different lock files,
# and write to different score logs.
#
# Active run resolution (Option B from the build spec): hard-coded
# $activeRun constant with an env-var override (V7_RUN_CONFIG). This is
# cleaner than threading a --version flag through build_public_dashboard
# right now because RUN_CONFIG doesn't yet have a v7 entry; when v7
# graduates into RUN_CONFIG we can swap to --print-active-run --version v7.
#
# Cadence: scans every 60s. Mirrors the v6 supervisor's WMI orphan-spawn
# pattern (Invoke-CimMethod Win32_Process Create with ShowWindow=0).

$pyEnv = 'C:\Users\cashc\Miniconda3\envs\image-gs'
$repo = 'E:\oss-gaussian-server'
$ckptRoot = 'E:\checkpoints'
$logFile = 'E:\logs\heldout-eval-v7-supervisor.log'

# v7 active run: env override wins so we can rotate without editing the
# script. The hard-coded default tracks the current Phase 3 run.
$activeRunDefault = 'srcnn-v7.0-pico-005'

$tartanairRoot = 'E:\datasets\tartanair_extracted'
$maxTriplets = 64
$seed = 42

function ResolveActiveRun {
    if ($env:V7_RUN_CONFIG) { return $env:V7_RUN_CONFIG.Trim() }
    return $activeRunDefault
}

function StepsAlreadyEvaluated {
    param([string]$scoreLog)
    # score_log_v7.json is a JSON ARRAY of eval rows (one per step). Parse
    # the whole file once and pull every step. Reading line-by-line as
    # JSONL is wrong and silently returns @() -- which would re-evaluate
    # every ckpt from scratch on every supervisor restart.
    if (-not (Test-Path $scoreLog)) { return @() }
    try {
        $raw = Get-Content $scoreLog -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($raw)) { return @() }
        $payload = ConvertFrom-Json $raw -ErrorAction Stop
        $steps = @()
        if ($payload -is [System.Collections.IEnumerable] -and -not ($payload -is [string])) {
            foreach ($row in $payload) {
                if ($row -ne $null -and $row.step -ne $null) { $steps += [int]$row.step }
            }
        } elseif ($payload -ne $null -and $payload.step -ne $null) {
            $steps += [int]$payload.step
        }
        return $steps
    } catch {
        return @()
    }
}

function StepFromCkptName {
    param([string]$name)
    if ($name -match 'step-(\d+)\.pt$') { return [int]$matches[1] }
    return -1
}

function Log {
    param([string]$msg)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $logFile -Value "[$stamp] $msg"
}

# v7-specific GPU mutex. SEPARATE from the v6 supervisor's lock so the two
# supervisors can target different runs without stomping on each other --
# but inside ONE supervisor's runs, this lock still serializes spawns to
# avoid the non-atomic check-then-spawn race.
$script:GpuLockPath = 'C:\temp\heldout-eval-v7.lock'
$script:GpuLockStaleSec = 7200  # 2h: longer than any realistic single eval.

function AcquireGpuLock {
    if (Test-Path $script:GpuLockPath) {
        $existing = Get-Item $script:GpuLockPath -ErrorAction SilentlyContinue
        if ($existing -and ((Get-Date) - $existing.LastWriteTime).TotalSeconds -gt $script:GpuLockStaleSec) {
            Log "removing stale GPU lock from $($existing.LastWriteTime)"
            Remove-Item $script:GpuLockPath -Force -ErrorAction SilentlyContinue
        }
    }
    try {
        $null = New-Item -ItemType File -Path $script:GpuLockPath -Value "$PID" -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function ReleaseGpuLock {
    Remove-Item $script:GpuLockPath -Force -ErrorAction SilentlyContinue
}

# AppendEvalToScoreLog: sr_eval_v7.py writes eval-step-NNNNNNNN.json to
# its --output-dir. The supervisor reads that file and appends the row to
# score_log_v7.json (dashboard polls this file). Dedupe by step.
function AppendEvalToScoreLog {
    param(
        [string]$scoreLog,
        [string]$evalJson,
        [int]$step
    )
    if (-not (Test-Path $evalJson)) {
        Log "eval json missing after spawn: $evalJson"
        return $false
    }
    try {
        $rowRaw = Get-Content $evalJson -Raw -ErrorAction Stop
        $row = ConvertFrom-Json $rowRaw -ErrorAction Stop
    } catch {
        Log "failed to parse eval json $evalJson : $_"
        return $false
    }
    # Ensure step field is present + matches the ckpt step.
    if ($row.PSObject.Properties.Name -notcontains 'step' -or $null -eq $row.step) {
        $row | Add-Member -NotePropertyName 'step' -NotePropertyValue $step -Force
    }
    # Load existing rows (or start empty).
    $rows = @()
    if (Test-Path $scoreLog) {
        try {
            $existingRaw = Get-Content $scoreLog -Raw -ErrorAction Stop
            if (-not [string]::IsNullOrWhiteSpace($existingRaw)) {
                $parsed = ConvertFrom-Json $existingRaw -ErrorAction Stop
                if ($parsed -is [System.Collections.IEnumerable] -and -not ($parsed -is [string])) {
                    $rows = @($parsed)
                } elseif ($parsed -ne $null) {
                    $rows = @($parsed)
                }
            }
        } catch {
            Log "failed to read existing $scoreLog, starting fresh: $_"
            $rows = @()
        }
    }
    # Drop any prior row at the same step (idempotency on retries) and
    # append the new one. Match the shape sr_eval_v7.py emits -- don't
    # transform; the dashboard reads the raw fields.
    $rows = @($rows | Where-Object { $_.step -ne $step })
    $rows += $row
    # Sort by step so downstream readers can rely on ascending order.
    $rows = @($rows | Sort-Object -Property step)
    try {
        $rows | ConvertTo-Json -Depth 16 | Set-Content -Path $scoreLog -Encoding UTF8
        return $true
    } catch {
        Log "failed to write $scoreLog : $_"
        return $false
    }
}

Log "v7 supervisor starting (active=$(ResolveActiveRun))"

while ($true) {
    $activeRun = ResolveActiveRun
    if (-not $activeRun) {
        Log "no active v7 run resolved; sleeping"
        Start-Sleep -Seconds 60
        continue
    }
    $runDir = "$ckptRoot\$activeRun"
    if (-not (Test-Path $runDir)) {
        Log "active=$activeRun but $runDir missing; sleeping"
        Start-Sleep -Seconds 60
        continue
    }

    $scoreLog = "$runDir\score_log_v7.json"
    $evalOutputDir = "$runDir\eval"
    if (-not (Test-Path $evalOutputDir)) {
        try { New-Item -ItemType Directory -Path $evalOutputDir -Force | Out-Null } catch {}
    }
    $evaluated = @(StepsAlreadyEvaluated -scoreLog $scoreLog)

    $ckpts = Get-ChildItem $runDir -Filter "step-*.pt" -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Where-Object {
            $s = StepFromCkptName -name $_.Name
            $s -gt 0 -and ($evaluated -notcontains $s)
        }

    if (-not $ckpts) {
        Start-Sleep -Seconds 60
        continue
    }

    # Skip eval while the trainer is currently writing a ckpt (avoid races
    # on partially-flushed files). Heuristic: skip if the file is < 60s old.
    $now = Get-Date
    $ready = $ckpts | Where-Object { ($now - $_.LastWriteTime).TotalSeconds -ge 60 }
    if (-not $ready) {
        Start-Sleep -Seconds 30
        continue
    }

    if (-not (AcquireGpuLock)) {
        Start-Sleep -Seconds 30
        continue
    }
    try {
        foreach ($ck in $ready) {
            $step = StepFromCkptName -name $ck.Name
            $stepPadded = $step.ToString('D8')
            $expectedEvalJson = "$evalOutputDir\eval-step-$stepPadded.json"
            Log "starting v7 eval: run=$activeRun step=$step ckpt=$($ck.FullName)"

            $argList = @(
                "scripts\sr_eval_v7.py",
                "--checkpoint", "`"$($ck.FullName)`"",
                "--tartanair-root", "`"$tartanairRoot`"",
                "--output-dir", "`"$evalOutputDir`"",
                "--device", "cuda",
                "--max-triplets", "$maxTriplets",
                "--seed", "$seed"
            ) -join ' '
            $cmdLine = "cmd /c `"cd /d $repo && $pyEnv\python.exe $argList >> E:\logs\heldout-eval-v7-$activeRun.log 2>&1`""

            # Hide the spawned eval console: SW_HIDE = 0
            $startupHidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
            $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine; ProcessStartupInformation = $startupHidden }
            if ($r.ReturnValue -ne 0) {
                Log "spawn FAILED step=$step rc=$($r.ReturnValue)"
                Start-Sleep -Seconds 30
                continue
            }
            Log "spawned v7 eval pid=$($r.ProcessId) step=$step"

            # Wait for THIS eval to finish before starting the next ckpt.
            # CIM transient failures fail closed (keep waiting) to avoid
            # racing the GPU.
            $running = $true
            while ($running) {
                Start-Sleep -Seconds 15
                try {
                    $still = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ProcessId)" -ErrorAction Stop
                    $running = [bool]$still
                } catch {
                    Log "CIM query transient failure waiting on pid=$($r.ProcessId); continuing wait"
                }
            }
            Log "completed v7 eval step=$step"

            # Append the eval json into score_log_v7.json so the dashboard
            # has a single rolled-up file to poll.
            if (AppendEvalToScoreLog -scoreLog $scoreLog -evalJson $expectedEvalJson -step $step) {
                Log "appended step=$step to $scoreLog"
            } else {
                Log "WARN: failed to append step=$step to $scoreLog"
            }

            # Re-read evaluated set for the next iteration.
            $evaluated = @(StepsAlreadyEvaluated -scoreLog $scoreLog)
        }
    } finally {
        ReleaseGpuLock
    }

    Start-Sleep -Seconds 60
}

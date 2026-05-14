# launch-v7-smoke.ps1 — one-shot v7 trainer smoke test with pass/fail validation.
#
# Phase 3 deliverable for v7-pico-005: wraps `scripts/sr_train_v7.py --steps 50`
# into a single command that exits 0 on a clean smoke or non-zero with a
# specific reason on failure. Designed to be the gatekeeper before kicking
# off the 100K-step pico-005 run under WMI orphan-spawn.
#
# Foreground execution (no hidden subprocess) — operator invokes this
# knowingly and wants to see live trainer output.
#
# Defaults match the pre-flight checklist in
# docs/architecture/2026-05-12-v7-pico-005-phase-3-plan.md § "Host wiring".
#
# Usage:
#   # Default smoke (50 steps, hat_tiny, max_triplets=8):
#   ssh 3080ti-windows powershell -NoProfile -ExecutionPolicy Bypass `
#     -File C:\Users\cashc\3080ti\launch-v7-smoke.ps1
#
#   # Override knobs:
#   powershell -File launch-v7-smoke.ps1 -Steps 100 -BackboneKind placeholder
#   powershell -File launch-v7-smoke.ps1 -OutputDir E:\checkpoints\v7-smoke-alt
#   powershell -File launch-v7-smoke.ps1 -Device cpu  # for CI-only sanity check
#
# Exit codes:
#   0  — all validations passed
#   2  — environment sanity (repo, python, tartanair) missing
#   3  — trainer ran but a validation check failed
#   non-zero (other) — trainer itself crashed; see smoke.log

param(
    [string]$Repo         = 'E:\oss-gaussian',
    [string]$PyEnv        = 'C:\Users\cashc\Miniconda3\envs\image-gs',
    [string]$TartanRoot   = 'E:\datasets\tartanair_extracted',
    [string]$OutputDir    = 'E:\checkpoints\srcnn-v7.0-smoke',
    [int]   $Steps        = 50,
    [int]   $BatchSize    = 2,
    [int]   $MaxTriplets  = 8,
    [string]$BackboneKind = 'hat_tiny',
    [string]$Device       = 'cuda'
)

function Run-Smoke {
    $ErrorActionPreference = 'Continue'
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $python = Join-Path $PyEnv 'python.exe'
    $trainer = Join-Path $Repo 'scripts\sr_train_v7.py'

    # ---- 1. Banner ------------------------------------------------------
    Write-Host '================================================================'
    Write-Host "  v7 SMOKE TEST — $ts"
    Write-Host '================================================================'
    Write-Host "  Repo         : $Repo"
    Write-Host "  PyEnv        : $PyEnv"
    Write-Host "  TartanRoot   : $TartanRoot"
    Write-Host "  OutputDir    : $OutputDir"
    Write-Host "  Steps        : $Steps"
    Write-Host "  BatchSize    : $BatchSize"
    Write-Host "  MaxTriplets  : $MaxTriplets"
    Write-Host "  BackboneKind : $BackboneKind"
    Write-Host "  Device       : $Device"
    Write-Host '----------------------------------------------------------------'

    # ---- 2. Sanity-check paths -----------------------------------------
    $missing = @()
    if (-not (Test-Path $Repo))       { $missing += "Repo: $Repo" }
    if (-not (Test-Path $python))     { $missing += "Python: $python" }
    if (-not (Test-Path $TartanRoot)) { $missing += "TartanRoot: $TartanRoot" }
    if (-not (Test-Path $trainer))    { $missing += "Trainer: $trainer" }
    if ($missing.Count -gt 0) {
        Write-Host ''
        Write-Host '[smoke FAIL] missing required paths:'
        foreach ($m in $missing) { Write-Host "  - $m" }
        exit 2
    }

    # ---- 3. Wipe + recreate OutputDir -----------------------------------
    if (Test-Path $OutputDir) {
        Write-Host "[smoke] clearing stale output: $OutputDir"
        Remove-Item -Recurse -Force $OutputDir
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $logPath     = Join-Path $OutputDir 'smoke.log'
    $historyPath = Join-Path $OutputDir 'history.jsonl'

    # ---- 4. Run trainer foreground, tee to smoke.log --------------------
    Write-Host "[smoke] launching trainer (tee -> $logPath)"
    Write-Host '----------------------------------------------------------------'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # Capture-and-stream: invoke trainer, tee output to log + console.
    & $python $trainer `
        --tartanair-root $TartanRoot `
        --output-dir $OutputDir `
        --steps $Steps `
        --batch-size $BatchSize `
        --max-triplets $MaxTriplets `
        --backbone-kind $BackboneKind `
        --device $Device `
        --log-every 10 `
        --ckpt-every $Steps `
        --curriculum 2>&1 | Tee-Object -FilePath $logPath

    $trainerExit = $LASTEXITCODE
    $sw.Stop()
    $wall = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    Write-Host '----------------------------------------------------------------'
    Write-Host "[smoke] trainer exited code=$trainerExit wall=${wall}s"

    # ---- 5. Validate ----------------------------------------------------
    $failures = @()

    # 5a. Exit code
    if ($trainerExit -eq 0) {
        Write-Host "[OK] trainer exit code = 0"
    } else {
        $failures += "trainer exit code = $trainerExit (expected 0)"
        Write-Host "[FAIL] trainer exit code = $trainerExit (expected 0)"
    }

    # 5b. history.jsonl row count
    $expectedRows = [math]::Ceiling($Steps / 10.0)
    $rows = @()
    if (Test-Path $historyPath) {
        $rows = Get-Content $historyPath | Where-Object { $_.Trim() -ne '' }
        if ($rows.Count -ge $expectedRows) {
            Write-Host "[OK] history.jsonl has $($rows.Count) rows (>= $expectedRows)"
        } else {
            $failures += "history.jsonl has $($rows.Count) rows, expected >= $expectedRows"
            Write-Host "[FAIL] history.jsonl has $($rows.Count) rows, expected >= $expectedRows"
        }
    } else {
        $failures += "history.jsonl missing at $historyPath"
        Write-Host "[FAIL] history.jsonl missing at $historyPath"
    }

    # 5c-e. Walk history rows for loss finiteness, canvas count, curriculum lambdas
    $lossFinite       = $true
    $canvasCountOk    = $true
    $curriculumOk     = $true
    $lastCanvasCount  = $null
    $lastOpacity      = $null
    $canvasCap        = 16384  # mirror trainer --canvas-capacity default (post-2026-05-13)
    $lossBadReason    = ''
    $canvasBadReason  = ''
    $currBadReason    = ''

    foreach ($line in $rows) {
        try {
            $r = $line | ConvertFrom-Json
        } catch {
            $lossFinite = $false
            $lossBadReason = "row not parseable as JSON: $line"
            continue
        }

        # total finite
        $tot = $r.total
        if ($null -eq $tot -or [double]::IsNaN([double]$tot) -or [double]::IsInfinity([double]$tot)) {
            $lossFinite = $false
            $lossBadReason = "row step=$($r.step) total=$tot (not finite)"
        }

        # canvas_count bounds
        $cc = $r.canvas_count
        if ($null -eq $cc -or [int]$cc -le 0) {
            $canvasCountOk = $false
            $canvasBadReason = "row step=$($r.step) canvas_count=$cc (must be > 0)"
        } elseif ([int]$cc -gt $canvasCap) {
            $canvasCountOk = $false
            $canvasBadReason = "row step=$($r.step) canvas_count=$cc > capacity=$canvasCap"
        } else {
            $lastCanvasCount = [int]$cc
        }

        # curriculum: stage-1 means lambda_fg must equal 0.0
        $lf = $r.lambda_fg
        if ($null -ne $lf -and [double]$lf -ne 0.0) {
            $curriculumOk = $false
            $currBadReason = "row step=$($r.step) lambda_fg=$lf (expected 0.0 in stage 1)"
        }

        if ($null -ne $r.canvas_mean_opacity) {
            $lastOpacity = [double]$r.canvas_mean_opacity
        }
    }

    if ($lossFinite) {
        Write-Host "[OK] all rows have finite total loss"
    } else {
        $failures += "loss finiteness: $lossBadReason"
        Write-Host "[FAIL] loss finiteness: $lossBadReason"
    }

    if ($canvasCountOk) {
        Write-Host "[OK] canvas_count in (0, $canvasCap] for all rows"
    } else {
        $failures += "canvas_count: $canvasBadReason"
        Write-Host "[FAIL] canvas_count: $canvasBadReason"
    }

    if ($curriculumOk) {
        Write-Host "[OK] stage-1 curriculum lambdas (lambda_fg=0.0)"
    } else {
        $failures += "curriculum: $currBadReason"
        Write-Host "[FAIL] curriculum: $currBadReason"
    }

    # 5f. Step-time floor (< 30 s/step). Production HAT-Tiny on 3080 Ti ~ 5 s.
    $perStep = if ($Steps -gt 0) { [math]::Round($wall / $Steps, 2) } else { 0 }
    if ($perStep -lt 30.0) {
        Write-Host "[OK] step time = ${perStep}s/step (< 30s floor)"
    } else {
        $failures += "step time = ${perStep}s/step (>= 30s floor)"
        Write-Host "[FAIL] step time = ${perStep}s/step (>= 30s floor)"
    }

    # 5g. Final checkpoint exists
    $finalCkpt = Get-ChildItem -Path $OutputDir -Filter 'step-*-final.pt' -ErrorAction SilentlyContinue
    if ($finalCkpt) {
        Write-Host "[OK] final ckpt present: $($finalCkpt[0].Name)"
    } else {
        $failures += "no step-*-final.pt in $OutputDir"
        Write-Host "[FAIL] no step-*-final.pt in $OutputDir"
    }

    # ---- 6. Summary ----------------------------------------------------
    Write-Host '----------------------------------------------------------------'
    if ($failures.Count -gt 0) {
        Write-Host "[smoke FAIL] $($failures.Count) check(s) failed:"
        foreach ($f in $failures) { Write-Host "  - $f" }
        Write-Host ''
        Write-Host "[smoke] last 20 lines of $logPath :"
        Get-Content $logPath -Tail 20 | ForEach-Object { Write-Host "  $_" }
        exit 3
    }

    $opStr = if ($null -ne $lastOpacity) { [math]::Round($lastOpacity, 4).ToString() } else { 'n/a' }
    $ccStr = if ($null -ne $lastCanvasCount) { $lastCanvasCount.ToString() } else { 'n/a' }
    Write-Host "[smoke PASS] $Steps steps in ${wall}s (${perStep}s/step) | final canvas count=$ccStr | mean_opacity=$opStr"
    exit 0
}

# Entry-point guard: skip execution when only parsing (for syntax validation).
if (-not $env:OSS_SMOKE_PARSE_ONLY) {
    Run-Smoke
}

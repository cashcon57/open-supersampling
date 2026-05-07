# Codex handoff — v6 in-flight viz

Date: 2026-05-06

## Change

- `scripts/sr_temporal_inflight_viz.py` now supports `--primary-version v6` and auto-detects `srcnn-v6-*` output dirs.
- v6-primary renders to `<output-dir>/viz/step-XXXXXXXX.png`, so the existing dashboard `/api/viz?run=srcnn-v6-heavy-001` route can list and serve the strips.
- v6-primary loads:
  - v6 from the watched checkpoint (`generator`, `model_state_dict`, or future `v6_config`/state keys);
  - required v5-pixel-temporal from `--ckpt-v5` or the validated-run fallback paths.
- The v6 strip layout is:
  `LR-bilinear | bicubic | v5-pixel-temporal | v6 | GT | |err v5| | |err v6|`
- CPU inference is forced and `torch.set_num_threads(2)` is applied.

## Verification

Passed:

```powershell
./venv-py312/bin/python -m pytest tests/test_sr_temporal_inflight_viz.py -q
```

Result: `3 passed`.

Full local suite was run as requested:

```powershell
./venv-py312/bin/python -m pytest tests/ -q
```

Result: `739 passed, 10 skipped, 3 failed`. The failures are existing environment/unrelated-suite failures:

- `tests/test_mitsuba_gen.py::test_zarr_schema`: local Mitsuba install reports missing `scale` plugin.
- `tests/test_runpod_client.py::{test_launch_no_orphan_raises,test_launch_empty_response_attempts_orphan_recovery}`: tests patch `ors.cloud.runpod_client...` while this repo imports `oss.cloud.runpod_client`.

## 3080ti Watcher Launch

Remote target: `<train-host>`

Command used for the orphan watcher:

```powershell
cd <train-host-data>\oss-gaussian
git fetch origin
git checkout v0.2-dev
git pull --ff-only origin v0.2-dev
New-Item -ItemType Directory -Force -Path <train-host-data>\checkpoints\srcnn-v6-heavy-001 | Out-Null

$cmd = 'cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_inflight_viz.py --output-dir <train-host-data>\checkpoints\srcnn-v6-heavy-001 --primary-version v6 --manifest <train-host-data>\checkpoints\v5_held_out_manifest.json --tartanair-root <train-host-data>\datasets\tartanair_extracted --interval 300 --n-pairs 4 > <train-host-data>\checkpoints\srcnn-v6-heavy-001\viz.log 2>&1'
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
}
```

Expected dashboard URL:

```text
http://localhost:8080/api/viz?run=srcnn-v6-heavy-001
```

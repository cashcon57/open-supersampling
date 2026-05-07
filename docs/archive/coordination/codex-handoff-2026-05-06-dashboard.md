# Codex Handoff — Dashboard Versioning

## Verification Blockers

Command run:

```bash
./venv-py312/bin/python -m pytest tests/ -m "not gpu and not mitsuba" --ignore=tests/gaussian -q
```

Result: 13 failed, 470 passed, 5 skipped, 2 deselected.

The failures appear outside the dashboard/viz feature scope:

- `tests/capture/test_e2e.py::{test_uploader_fake_server_roundtrip_deletes_terminal_and_exhausted_frames,test_pending_cap_evicts_oldest_pairs_before_upload}` fail because the sandbox denies localhost socket binding with `PermissionError: [Errno 1] Operation not permitted`.
- `tests/sr/v6/test_dataset.py::test_sr_train_v6_smoke_passes_construction` times out after 120s while running `scripts/sr_train_v6.py --smoke`. `scripts/sr_train_v6.py` is already dirty from other work in this checkout and is unrelated to this dashboard task.
- `tests/test_onnx_parity.py::test_onnx_export_round_trip` fails ONNX parity at RGB max diff `0.00244140625` vs threshold `< 0.002`.
- `tests/test_runpod_client.py::{test_launch_no_orphan_raises,test_launch_empty_response_attempts_orphan_recovery}` fail with `ModuleNotFoundError: No module named 'ors'` from `<home>/open-reconstruction-suite/tests`.
- `tests/test_safety_harness.py` safety harness tests fail because the sandbox denies writes to `<home>/.ors-cloud-heartbeats/*.beat` and `<home>/.ors-cloud-audit.log`.

Feature-local checks passed:

```bash
./venv-py312/bin/python -m pytest tests/test_dashboard_versioning.py tests/test_sr_temporal_inflight_viz.py -q
```

Result: 5 passed.

# CI Heal Stuck: fa01a6e

Date: 2026-05-07

## Original failure

Commit `fa01a6e` failed the CPU smoke job in the broader test slice:

```text
FAILED tests/capture/test_uploader.py::test_enforce_pending_cap_deletes_oldest_pair
assert deleted == [oldest.frame_path]
actual deleted path: newest.exr
```

The pending-cap eviction code used filesystem timestamp ordering and then the full path as a tie-breaker. On GitHub Actions, the test-created files landed with colliding timestamp order, so the random UUID session directory could decide the eviction order and delete `newest.exr`.

## Fix attempted

Commit `00c4035` (`ci: fix pending cap eviction ordering (re fa01a6e)`) changed `oss/capture/uploader.py` so pending-cap eviction:

- uses `captured_at_unix` from capture metadata as the primary age signal;
- falls back to nanosecond mtime/ctime ordering when metadata is unavailable or tied;
- only uses frame name/full path as the final deterministic tie-breaker.

Local verification passed:

```text
./venv-py312/bin/python -m pytest tests/capture/test_uploader.py -q --tb=short
9 passed in 0.07s

./venv-py312/bin/python -m pytest tests/capture -q --tb=short
78 passed, 1 skipped in 3.52s
```

## CI result after fix

Run `25510426850` for `00c4035` failed, but not on the original pending-cap assertion. The broader slice progressed past that area and failed later:

```text
FAILED tests/test_onnx_parity.py::test_onnx_export_round_trip
AssertionError: RGB parity failed: 0.005859375
threshold: rgb_diff < 2e-3
hidden diff: 2.957880e-06
```

The new red is an ONNX parity tolerance/input-stability issue in `oss/export/onnx_export.py` or its fixture path, not the original pending-cap eviction failure. Other auto-heal commits were already landing on `main` for ONNX parity while this run was being monitored, so I stopped here per the CI-heal instruction to write a stuck memo when the monitored run remained red.

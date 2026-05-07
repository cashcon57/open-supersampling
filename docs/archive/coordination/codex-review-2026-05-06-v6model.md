# Codex review: V6Model orchestrator audit 2026-05-06

## Findings

- Severity: HIGH
  File: `oss/sr/v6/model.py:234`
  Description: Non-empty canvas forward crashes before cross-attention. `_build_canvas_tokens()` calls `KeyframeActiveMaskCache.get_mask(..., view_matrix=None)`, but `_compute_active_mask()` immediately calls `view_matrix.to(...)`; the first canvas-backed frame raises `AttributeError: 'NoneType' object has no attribute 'to'`. This means the only working path is the empty-canvas K=0 path.
  Suggested fix: Pass an explicit identity affine matrix on the canvas device/dtype, plus a real viewport size from `feats.shape[-2:]` or `canvas.output_hw`; alternatively update `KeyframeActiveMaskCache` to treat `None` as identity and require/derive `viewport_hw`.

- Severity: MEDIUM
  File: `oss/sr/v6/model.py:172`
  Description: `frame_index` is accepted by `forward()` but ignored; `_build_canvas_tokens()` keys the active-mask cache with `_step_count` instead. If the trainer calls `forward()` across frames without calling `maybe_prune()` between them, every non-empty forward is treated as frame 0. If `maybe_prune()` is called once per optimizer step, keyframe-mask recomputation is driven by training-step cadence rather than frame cadence, silently selecting stale or wrong active Gaussian sets.
  Suggested fix: Thread `frame_index` into `_build_canvas_tokens(frame_index=frame_index)` and pass it to `keyframe_mask.get_mask()`. Keep prune-step accounting separate from frame/keyframe accounting.

- Severity: MEDIUM
  File: `oss/sr/v6/model.py:137`
  Description: `_step_count` is local mutable scheduler state, but it is neither reset by `reset_state()` nor saved in `state_dict()`. Reusing a model object for a fresh training run inherits the old prune cadence; checkpoint/resume loses the old cadence and can skip or delay a prune boundary, e.g. saving at step 199 with `prune_every=200` resumes at local step 0 instead of pruning next call.
  Suggested fix: Define one owner for prune step state. Prefer `maybe_prune(step: int)` driven by the trainer's checkpointed global step, or register/save a buffer if the model owns it. If `reset_state()` is meant to start a fresh run, add an explicit option to reset `_step_count`.

- Severity: LOW
  File: `tests/sr/v6/test_model.py:112`
  Description: The gradient-flow test only covers K=0 and explicitly filters out `fusion.*` and `canvas_to_token.*`, so it cannot catch broken non-empty canvas attention, token projection, or backward flow.
  Suggested fix: Add a non-empty synthetic `CanvasState` forward/backward test with valid `output_hw`/identity view, assert output shape/finite loss, and assert gradients reach `canvas_to_token` and the cross-attention projections.

- Severity: LOW
  File: `tests/sr/v6/test_model.py:168`
  Description: The prune test reaches the `prune_every` boundary only with empty canvas and no `STVScoreState`, so it does not prove exact-boundary pruning, returned prune count, or canvas/ST state shrinking.
  Suggested fix: Inject a 4- or 8-Gaussian `CanvasState` plus matching `STVScoreState`, set `prune_every` small, call `maybe_prune()` through exactly the boundary, and assert no early prune, expected count at the boundary, and matching reduced tensor/state shapes.

- Severity: LOW
  File: `tests/sr/v6/test_model.py:1`
  Description: Missing save/load coverage for orchestrator behavior that is not in `state_dict()` (`V6Config`, `color_activation`, `_step_count`, and local canvas/ST/keyframe state). This leaves checkpoint-resume behavior undefined.
  Suggested fix: Add a state_dict round-trip test with same config for numerical parity on empty-canvas forward, and a separate test documenting that local canvas/ST/keyframe state is intentionally not serialized. If prune cadence remains model-owned, include `_step_count` in the round trip.

## Notes

- Empty-canvas frame 0 K=0 path looks correct: `V6Model` builds `(B, 0, token_dim)` tokens, and `PixelGaussianFusion.forward()` returns `pixel_features` unchanged when `k == 0`.
- Color activation switch looks correct: `softplus` is the default HDR-capable non-negative path; `sigmoid` is opt-in and clamps to `[0, 1]`.
- Requested test command: `./venv-py312/bin/python -m pytest tests/sr/v6/ -q` passed with `184 passed, 9 warnings`.

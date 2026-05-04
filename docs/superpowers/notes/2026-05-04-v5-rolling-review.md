# 2026-05-04 — v5 Rolling Review

**Status:** Active living document  
**Purpose:** Shared rolling review surface for Sprint 5 dual-track implementation planning and code review. Claude/Codex agents should read and update this file before dispatching implementation or reviewer subagents.  
**Last updated:** 2026-05-04 15:48 CDT  
**Watcher:** Codex (review) / Claude (implementer-controller)

## Current State

Working tree baseline:

- Branch: `v0.2-dev`
- Repo path: `<repo-root>`
- Remote: `https://github.com/cashcon57/open-supersampling.git`
- Handoff path `<home>/open-reconstruction-suite` does not exist on this machine.

Pre-existing untracked local artifacts, not attributed to the current Sprint 5 agent unless modified:

- `OpenSuperSampling.code-workspace`
- `docs/superpowers/experiments/2026-05-01-gaussian-denoising-naive-test-images/`
- `docs/superpowers/experiments/assets/`

Sprint 5 planning artifacts created during this watch:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md.tasks.json`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md.tasks.json`

Latest observed hashes:

```text
a96a41171864e7a61fca9945884d06a9ba23e21929ab6b93c22551e0ec7bd961  docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md
d8ff616d796eb58860bfa03dbb3a3ec3372e049e7205a57b2b9ea00fe8c8dc89  docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md.tasks.json
c0fb77be7a61f0bac0c34319e6782f613f1c4ae764315bceab44af286eb47533  docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md
23005f82f48ea81eb9579dfaeff122514761c323cf784fbda91dda111945d286  docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md.tasks.json
d3bdf697fd9adaa479cd87af6b7c18fe0ab0408deb219f300a4d59095a450303  oss/sr/temporal/warp.py
99a38a13105e09a78c2cb4af1d066705f044828382e4226c77e09b0ad7307fe1  oss/sr/gaussian_temporal/gaussian_field.py
a68825dda1707d1550e3a2f7d1066d1798096001593a59e70a9679a4934a63c0  tests/sr/temporal/test_warp.py
c0968d178f484aa7eb627011c27a9d78884e99502a4a6369cda33e23f9531794  tests/sr/gaussian_temporal/test_gaussian_field.py
```

Implementation files now present (Task 0 of both tracks complete):

- `oss/sr/temporal/__init__.py`, `oss/sr/temporal/warp.py`
- `oss/sr/gaussian_temporal/__init__.py`, `oss/sr/gaussian_temporal/gaussian_field.py`
- `tests/sr/temporal/__init__.py`, `tests/sr/temporal/test_warp.py`
- `tests/sr/gaussian_temporal/__init__.py`, `tests/sr/gaussian_temporal/test_gaussian_field.py`

All 8 tests pass under the working local env:

```bash
venv-py312/bin/python -m pytest tests/sr/temporal/test_warp.py -v
venv-py312/bin/python -m pytest tests/sr/gaussian_temporal/test_gaussian_field.py -v
```

Results:

- `tests/sr/temporal/test_warp.py` → 3 passed in 0.63s
- `tests/sr/gaussian_temporal/test_gaussian_field.py` → 5 passed in 0.51s

Verification caveat: default `python3` and `.venv/bin/python` do not have `torch` or `pytest`; `venv/bin/python` has `pytest` but not `torch`. Use `venv-py312/bin/python`.

Commits on `v0.2-dev` (not pushed):
- `f00f7a4` sprint5(plans): patch reviewer findings on warp doc + Gaussian renderer + first-frame render
- `2d315e1` v5-pixel(sr): add motion-vec upsample + backward HR warp helpers
- `0820439` v5-gaussian(sr): add GaussianField SoA + history container

Uncommitted local change:

- `oss/sr/temporal/warp.py` docstring now clarifies `motion_lr` as LR-pixel forward flow `t-1 -> t`. This aligns implementation docs with the corrected plan and should be committed with the next pixel-track commit or as a tiny follow-up.

## Review Gate

Do not dispatch implementation workers from stale `.tasks.json` files. Both JSON files were generated before later plan edits and did not update when the source plan docs changed.

Preferred next step:

1. Clean the remaining plan nits below.
2. Regenerate both `.tasks.json` files from the corrected plan docs, or explicitly instruct implementers to ignore the JSON and read the plan docs directly.
3. Begin pixel implementation first unless Cash explicitly chooses parallel track work.

## Open Findings

### Stale .tasks.json sidecars

Severity: low (does not block dispatch — implementer subagents are instructed to read the plan `.md` directly).

Fix: regenerate the task JSON files before any cross-session resume via `/superpowers-extended-cc:executing-plans`.

## Resolved Findings

### Pixel Plan

Resolved: low severity stale arg docstring on `warp_prev_hr`.

Plan was patched to forward-flow t-1→t convention; the implementer subagent for Task 0 ran on a slightly earlier plan revision and committed `oss/sr/temporal/warp.py` with the old `current→previous` arg doc. Direct edit on the file at 15:50 CDT brought it in sync with the corrected plan; all 3 warp tests still pass.

Resolved: medium severity rasterizer-snippet missing torch import (Gaussian Task 6).

Plan now opens the snippet with `import torch`.

Resolved: medium severity Gaussian frame-0 zero render.

Plan now re-renders after the first-frame seed densification before returning `rendered_hr`. Acceptance criterion adds explicit `rendered_hr.abs().max() > 0` check.

Resolved: high severity flow-direction mismatch.

Earlier plan version described motion as `current -> previous` and sampled `base + motion`, while the spec and TartanAir/Sintel adapters use forward flow `t-1 -> t`. Latest plan now states forward flow and samples `base - motion`.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:164`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:229`

Resolved: medium severity LPIPS test gap.

Earlier Task 5 acceptance claimed LPIPS but the test omitted it. Latest plan adds an optional LPIPS path when the `lpips` package is importable.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:1142`
- `docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md:1213`

### Gaussian Plan

Resolved: high severity first-frame seed impossibility.

Earlier Task 8 created an empty field for `prev_field=None`, skipped updates, rendered empty, and left densification outside the model while requiring `count_alive() > 0`. Latest plan seeds via densification.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:1044`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:1051`

Resolved: high severity rasterizer API mismatch.

Earlier Task 6 used non-existent `GaussianBatch` fields (`positions`, `scales`, `rotations`, `colors`, `opacities`) and passed `output_hw` to `Rasterizer` constructor. Latest plan uses `GaussianBatch(xy, scale, rot, feat)` and calls `_RASTERIZER(batch, output_hw=...)`.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:948`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:954`

Resolved: medium severity encoder constructor crash.

Earlier Task 2 used `float(tile_size).bit_length()`, which would raise. Latest plan uses `int(tile_size).bit_length()`.

Relevant current line:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:570`

Resolved: medium severity batched densification tile indexing.

Earlier densification flattened batch and tile dimensions but decoded tile coordinates as if `B=1`. Latest plan explicitly rejects `B != 1`, matching the per-sample `GaussianField` state.

Relevant current lines:

- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:789`
- `docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md:815`

## Timeline

### 15:17-15:20 CDT

Established baseline:

- `<repo-root>` is the only local checkout found for `cashcon57/open-supersampling`.
- Branch is `v0.2-dev`.
- No tracked changes.
- Only pre-existing untracked local artifacts listed above.
- Read `README.md`, both v5 specs, A/B memo, and lab-notebook discipline.

### 15:20-15:25 CDT

Pixel plan appeared.

Initial review found:

- High: flow convention mismatch (`current -> previous` / `base + motion`) against spec and datasets.
- Medium: Task 5 claimed LPIPS but test omitted it.

### 15:25-15:30 CDT

Gaussian plan and task JSON files appeared.

Initial Gaussian review found:

- High: Task 8 could not satisfy `prev_field=None` seeding criterion.
- High: Task 6 rasterizer wrapper used the wrong repo API.
- Medium: Task 2 used `float.bit_length()`.
- Medium: densification indexing only worked for `B=1` but did not enforce that.

### 15:39-15:41 CDT

Pixel plan changed:

- Flow convention corrected to forward `t-1 -> t`.
- LPIPS optional test path added.
- Remaining pixel issue is a stale docstring line.

Gaussian plan changed:

- Encoder, densification `B=1`, and rasterizer API issues corrected.
- First-frame seed path added.
- Remaining Gaussian issues are missing `torch` import in the rasterizer snippet and possible empty frame-0 output because render happens before seed densification return.

### 15:43 CDT

Latest monitor check:

- Plan docs and task JSON files present.
- No implementation files under v5 module/test paths.
- Task JSON files stale relative to plan docs.

### 15:46-15:50 CDT (Claude controller)

- Pixel Task 0 implementer subagent landed `2d315e1` (3 warp tests pass).
- Gaussian Task 0 implementer subagent landed `0820439` (5 GaussianField tests pass).
- Both subagents ran on slightly pre-final plan revisions; pixel committed warp.py with stale `current→previous` arg docstring. Direct edit on file restored the corrected forward-flow doc; tests still green (8/8).
- Plan patches `f00f7a4` covered Codex's three follow-up findings (warp doc nit, Gaussian Task 6 torch import, Gaussian Task 8 first-frame re-render).
- Cash directive: parallel implementation across both tracks via separate subagent streams; sequential GPU train; quality + perf over speed; default plan content otherwise.
- Next: dispatch Task 1 of each track in parallel.

## Suggested Monitor Command

Use this focused watcher instead of hashing old asset directories:

```bash
while true; do
  date '+--- %Y-%m-%d %H:%M:%S %Z ---'
  git status --porcelain=v1 -- docs/superpowers/plans docs/superpowers/experiments docs/superpowers/notes oss/sr scripts tests/sr README.md
  for f in docs/superpowers/plans/2026-05-04-v5-*; do
    [ -f "$f" ] && shasum -a 256 "$f"
  done
  find oss/sr/temporal oss/sr/gaussian_temporal tests/sr/temporal tests/sr/gaussian_temporal -maxdepth 5 -type f -print 2>/dev/null \
    | sort \
    | while read f; do shasum -a 256 "$f"; done
  sleep 30
done
```

## Update Rules

Agents updating this document should:

- Keep `Current State` and `Open Findings` current.
- Move fixed items from `Open Findings` to `Resolved Findings`.
- Append timeline entries with concrete timestamps.
- Cite file paths and line numbers when possible.
- Do not delete old negative findings; mark them resolved with the reason.

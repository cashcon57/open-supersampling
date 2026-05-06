# Codex handoff: v6 Stage 2 model wire-up

Date: 2026-05-06

Local commit created:

`fd8965f16e6ce8182ee06d86e3e318178331b58e`

Commit title:

`v6(model): canonical-memo Stage 2 — canvas warp + rasterizer + write-back in HR critical path`

Push status:

`git push origin v0.2-dev` failed in the sandbox because DNS resolution for `github.com` is blocked:

```text
fatal: unable to access 'https://github.com/cashcon57/open-supersampling.git/': Could not resolve host: github.com
```

Changed files:

- `<repo-root>/oss/sr/v6/model.py`
- `<repo-root>/oss/sr/v6/gaussian_spawner.py`
- `<repo-root>/tests/sr/v6/test_model.py`

Diff stat versus `origin/v0.2-dev`:

```text
 oss/sr/v6/gaussian_spawner.py |   5 +-
 oss/sr/v6/model.py            | 276 +++++++++++++++++++++++++++++++++++++++---
 tests/sr/v6/test_model.py     | 114 ++++++++++++++++-
 3 files changed, 371 insertions(+), 24 deletions(-)
```

Verification:

```text
./venv-py312/bin/python -m pytest tests/sr/v6/test_model.py -q
23 passed in 3.31s

./venv-py312/bin/python -m pytest tests/sr/v6/test_gaussian_spawner.py tests/sr/v6/test_canvas_warp.py tests/sr/v6/test_v6_rasterizer.py -q
31 passed in 0.49s

./venv-py312/bin/python -m pytest tests/sr/v6/ -q
234 passed, 10 warnings in 15.93s
```

Notes:

- `V6Model.forward()` now follows the canonical Stage 2 path: HAT LR features, optional canvas warp, active-mask token fusion, post-fusion Gaussian spawning, flattened per-rank canvas write-back, active-subset HR rasterization, refined-feature HR interpolation, composite RGB head, output activation, and ST-score state update.
- Batched `GaussianSpawnState` is flattened into unbatched `CanvasState` before rasterizer/warp use. The persistent canvas is bounded by `canvas_capacity` by dropping oldest entries.
- `V6Config.tile_size_hr` was added with default `16` and is passed to `V6Rasterizer`; `tile_size_lr` continues to configure `GaussianSpawner`.
- `GaussianSpawner` now initializes color embedding bias to `0.01` so fresh spawned canvas tokens produce nonzero `canvas_to_token.weight` gradient while keeping geometry neutral.

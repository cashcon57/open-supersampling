# Codex handoff — v6 Gaussian spawner

Date: 2026-05-06

## Status

Implemented and tested `oss/sr/v6/gaussian_spawner.py`.

Local `HEAD` and local `origin/v0.2-dev` both point at:

`8c847279981ca45a2248e775340e1bf541e2829d`

Commit subject currently present locally:

`v6 stage 2 phase 1: spawner + canvas_warp + rasterizer building blocks`

Codex attempted `git push origin v0.2-dev`, but network was blocked:

`fatal: unable to access 'https://github.com/cashcon57/open-supersampling.git/': Could not resolve host: github.com`

Please verify GitHub has commit `8c847279981ca45a2248e775340e1bf541e2829d`. If it does not, push `v0.2-dev` from an environment with network access.

## Spawner Changes

- Added `GaussianSpawner`, a GRAPE-style point-wise decoder:
  - one `nn.Conv2d(feat_dim, 6 + token_dim, kernel_size=1)`
  - per-pixel params pooled to LR tiles with `avg_pool2d`
  - outputs batched `GaussianSpawnState`
- Output fields:
  - `positions`: `(B, K, 2)` HR pixel-space tile centers plus bounded learned offsets
  - `scales`: `(B, K, 2)` via `softplus`, initialized near `0.5 * tile_size_hr`
  - `rotations`: `(B, K)` via `tanh * pi`
  - `colors`: `(B, K, token_dim)`
  - `confidence`: `(B, K)` via `sigmoid`
  - `opacities` property aliases `confidence` for CanvasState-style consumers
- Added `V6Config.tile_size_lr = 8`.
- Exported `GaussianSpawner` and `GaussianSpawnState` from `oss.sr.v6`.
- Added `tests/sr/v6/test_gaussian_spawner.py`.

## Verification

Commands run:

```bash
./venv-py312/bin/python -m pytest tests/sr/v6/test_gaussian_spawner.py -q
./venv-py312/bin/python -m pytest tests/sr/v6/ -q
```

Results:

- `tests/sr/v6/test_gaussian_spawner.py`: `19 passed`
- Full v6 suite: `230 passed, 10 warnings`

Warnings were existing torchvision / pytorch_wavelets deprecations plus the existing DDP smoke skip when CUDA and loopback bind are unavailable.

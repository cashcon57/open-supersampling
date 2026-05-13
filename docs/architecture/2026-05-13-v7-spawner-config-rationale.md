# v7 spawner config rationale — bench-data-driven defaults

**Date:** 2026-05-13
**Status:** New defaults in `V7Config` apply to all v7 training + inference. Backed by `/tmp/bench_canvas.py` and `/tmp/bench_deployment_res.py` measurements on a CPU host (cashs-macbook-pro). GPU numbers will be ~30-50x faster; relative orderings hold.

## What changed

| Setting | Old default | New default | Why |
|---|---|---|---|
| `canvas_capacity` | 4096 | **16384** | Old default overflowed on **first spawn** at TartanAir's native 480x640 HR (4800 > 4096). New default holds 2 spawns of 2400 = 4800 actives at TartanAir HR with 3.4x headroom. |
| `spawner_k_per_tile` | 4 | **2** | k=4 produced 4800 Gaussians per spawn at TartanAir HR which doesn't fit the (old or new) capacity. k=2 → 2400/spawn fits the new capacity comfortably and leaves room for the parent-child mechanism to add ~10000 more actives where loss is high. |
| `spawner_tile_size` | 16 | 16 (unchanged) | tile=16 matches v6.x HAT-Tiny window_size + makes 1080p HR pad to one extra tile, not many. tile=8 would be sub-pixel-friendlier but spawn count quadruples; tile=32 is too coarse (1 Gaussian / 64 HR pixels insufficient for fine geometry). |
| Spawner forward at non-divisible HR | ValueError | reflect-pad to nearest tile boundary | Deployment HR shapes (720p, 1080p, 1440p) are not always divisible by 16. Padding lets the trainer + inference run at arbitrary HR; padded-tile Gaussians fall slightly outside the HR rect but the rasterizer culls them at composite time. |

## Measurements (CPU; relative orderings hold on GPU)

### Spawn density at TartanAir HR (480x640)

| tile | k_per_tile | Gaussians/spawn | 2-spawn total | Fits cap=16384? |
|---|---|---|---|---|
| 32 | 2 | 600 | 1200 | yes (under-dense) |
| 16 | 2 | **2400** | **4800** | **yes (chosen)** |
| 16 | 4 | 4800 | 9600 | tight |
| 8 | 2 | 9600 | 19200 | no |

### Spawn density at 1080p HR (1080x1920, the most common deployment target)

| tile | k_per_tile | Gaussians/spawn | 2-spawn total | Capacity required |
|---|---|---|---|---|
| 32 | 2 | 4080 | 8160 | cap ≥ 8192 |
| **16** | **2** | **16320** | **32640** | **cap ≥ 65536** |
| 8 | 2 | 64800 | 129600 | cap ≥ 131072 |

### Rasterizer wall-time (CPU) vs n_active at 480x640 HR

| n_active | wall_ms (CPU) | est wall_ms (3080 Ti) |
|---|---|---|
| 256 | 14 | ~0.5 |
| 1024 | 51 | ~1.7 |
| 4096 | 206 | ~7 |
| 8192 | 408 | ~14 |
| 16384 | 803 | ~27 |
| 32768 | 1625 | ~55 |

The trainer does 3 renders per sample (frame N at t=0, frame N+1 at t=2, intermediate at t=1) plus backward through them. At pico-005's planned 4800-active 2-spawn cycle, total render time is ~21 ms × 3 = ~63 ms on GPU; backward roughly 2-3x forward → ~200 ms/sample. With B=2 → ~400 ms/step from rasterizer alone, plus backbone forward+backward. Comports with the Phase 3 plan's 4.5-7.0 s/step estimate.

## Capacity sizing rule for deployment

For an HR shape `(H, W)` with `tile_size=T, k_per_tile=K`:

```
spawn_count = ceil(H/T) * ceil(W/T) * K
canvas_capacity_minimum = 2 * spawn_count       # 2 spawns per trainer step
canvas_capacity_recommended = 4 * spawn_count   # leaves headroom for parent-child
                                                 # materializations
```

Concrete recommendations:

| Scenario | HR | Recommended `canvas_capacity` |
|---|---|---|
| TartanAir training (480x640) | 480x640 | **16384** (default, fits with 3.4x headroom) |
| 240p -> 720p inference | 720x1280 | 16384 |
| 360p -> 1080p inference | 1080x1920 | **65536** |
| 540p -> 1080p inference | 1080x1920 | **65536** |
| 720p -> 1440p inference | 1440x2560 | **131072** |
| 1080p -> 4K inference | 2160x3840 | **131072** (with k=1 or k=2 + tile=32) |

For deployment beyond TartanAir's training HR, users override via `--canvas-capacity` (training) or the V7Config field (inference).

## What this does NOT solve

- **Sub-pixel feature density.** Uniform per-tile spawning at k=2 means ~1 Gaussian per 128 HR pixels (16x16 / 2). Thin wires + chain-link fences + hair will alias. Mitigation paths:
  - Parent-child spawner integration (see `2026-05-13-v7-parent-child-integration-debt.md`).
  - Sobel high-frequency loss term (`--lambda-sobel 0.1`, off by default; landed in same commit).
- **Memory at extreme HR.** 4K HR with k=2 and tile=16 needs ~130k Gaussians; canvas_capacity=131072 occupies ~10 MB of float32 storage on its own. Larger feature_dim multiplies that. For 4K shipping, the Pico student model will use a smaller latent_rank and possibly tile=32 to halve the count.
- **Render time at extreme HR.** The current pure-Python rasterizer is too slow for real-time at 4K HR. CUDA kernel port (Phase 4 in v7 spec) addresses this. For training-time eval the CPU/GPU eager-mode rasterizer is fine.

## Test coverage

- `tests/sr/v7/test_spawner_resolution.py` — 8 tests covering non-divisible HR, padding-no-NaN, 2-spawn-cycle fit at deployment resolutions, default config fits TartanAir.
- `tests/sr/v7/test_sobel_loss.py` — 5 tests for the new Sobel HF loss term.
- All other v7 tests still pass; new defaults didn't regress existing behavior.

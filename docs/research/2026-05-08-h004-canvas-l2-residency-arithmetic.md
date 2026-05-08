# H004 Canvas L2 Residency Arithmetic

Date: 2026-05-08

## Verdict

Refute the claim as stated. The repo does not currently implement an exact
fp16-packed persistent canvas struct with `xy + conic/scale-rot + z + confidence
+ age + cov_id`.

The current v6 persistent canvas is a PyTorch SoA `CanvasState` with fp32
`positions`, fp32 `scales`, fp32 `rotations`, fp32 `opacities`, and fp32
`colors`; its raw storage is `24 + 4R` bytes/Gaussian before the separate
ST-score state. With the ST-score age-like state included, it is `36 + 4R`
bytes/Gaussian. At `R=64`, that is about 4.4-4.6 MiB for 16K Gaussians, close
enough to a 6 MiB 3080 Ti L2 that pair lists, conic scratch, output tiles, and
model activations will not all be L2-resident.

The 32 bytes/Gaussian number is still plausible as a future packed fp16 low-rank
runtime record for `R=4` or `R=8`, but not for `R=64`: even an ideal 16-byte
aligned fp16 record with the requested metadata is 144 bytes/Gaussian at `R=64`.

## Repo Evidence

- `oss/sr/v6/model.py` defines `CanvasState` as `positions`, `scales`,
  `rotations`, `opacities`, `colors`, and `count`; no `age` or `cov_id` field is
  in the canvas record.
- `oss/sr/v6/gaussian_spawner.py` decodes `positions`, `scales`, `rotations`,
  `colors`, and `confidence`; `confidence` is exposed to canvas consumers as
  `opacities`.
- `oss/sr/v6/rasterizer.py` supports `latent_rank < 64` by truncating/padding
  `canvas.colors` for raster output, but the stored canvas tensor remains the
  `colors` tensor provided by `CanvasState`.
- `oss/sr/v6/st_variation_score.py` stores lifespan separately as int64
  `lifespan_count`, plus fp32 `spatial_accumulator`; this is the closest current
  implementation to per-Gaussian age.
- `oss/gaussian/canvas/canvas.py` has an older `PersistentCanvas` SoA with
  `positions`, `scales`, `rotations`, `colors`, `age`, `error`, and `alive`; it
  has no confidence/opacities or `cov_id`.
- `oss/cuda/src/rasterizer_fwd.cu` takes fp32 SoA tensors and creates fp32
  transient conic scratch. It is not a persistent packed canvas layout.
- The Metal and Vulkan scaffold structs use fp32 AoS records with `xy`, `scale`,
  `rot`, padding, and `feat4`. Due to `float4` alignment, that record is 48
  bytes/Gaussian, not 32, and it also lacks `age` and `cov_id`.

## Arithmetic Assumptions

All totals use `N = 16K = 16,384` Gaussians and binary MiB. L2 comparisons use
the requested capacities: 3080 Ti L2 = 6 MiB, 4070 mobile L2 = 32 MiB.

For the hypothetical packed layout, use:

| Field | Bytes |
| --- | ---: |
| `xy` fp16x2 | 4 |
| conic fp16x3, or scale fp16x2 + rot fp16x1 | 6 |
| `z`/`feat` fp16xR | `2R` |
| confidence fp16 | 2 |
| age uint16 | 2 |
| `cov_id` uint16 | 2 |
| Raw subtotal | `16 + 2R` |
| Aligned record | `ceil((16 + 2R) / 16) * 16` |

## Packed fp16 Candidate

| R | Raw B/Gaussian | 16B-aligned B/Gaussian | N=16K total | 3080 Ti 6 MiB | 4070 mobile 32 MiB | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 24 | 32 | 512 KiB / 0.50 MiB | 8.3% | 1.6% | 32 B claim holds after padding |
| 8 | 32 | 32 | 512 KiB / 0.50 MiB | 8.3% | 1.6% | 32 B claim holds exactly |
| 64 | 144 | 144 | 2304 KiB / 2.25 MiB | 37.5% | 7.0% | 32 B claim fails |

This table is the favorable case for the claim. It assumes no fp32 persistent
geometry, no per-tensor storage overhead, no active mask, no tile index lists,
and no output/intermediate working set.

## Closest Implemented v6 State

Current v6 `CanvasState` raw tensor storage:

`positions fp32x2 + scales fp32x2 + rotations fp32 + opacities fp32 + colors fp32xR`
= `24 + 4R` bytes/Gaussian.

With current ST-score state included:

`CanvasState + spatial_accumulator fp32 + lifespan_count int64`
= `36 + 4R` bytes/Gaussian.

| R | CanvasState B/Gaussian | CanvasState total | With ST-score B/Gaussian | With ST-score total | 3080 Ti 6 MiB, with ST | 4070 mobile 32 MiB, with ST |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 40 | 640 KiB / 0.625 MiB | 52 | 832 KiB / 0.813 MiB | 13.5% | 2.5% |
| 8 | 56 | 896 KiB / 0.875 MiB | 68 | 1088 KiB / 1.063 MiB | 17.7% | 3.3% |
| 64 | 280 | 4480 KiB / 4.375 MiB | 292 | 4672 KiB / 4.563 MiB | 76.0% | 14.3% |

## Other Implemented Layouts

| Layout | Fields counted | Bytes/Gaussian | N=16K total | Notes |
| --- | --- | ---: | ---: | --- |
| v5 `PersistentCanvas`, raw SoA, `F=64` | fp32 xy, scale, rot, color; int64 age; fp32 error; bool alive | 289 | 4624 KiB / 4.516 MiB | No confidence or `cov_id`; `error` is not confidence |
| Metal/Vulkan scaffold `Gaussian` | fp32 `xy`, fp32 `scale`, fp32 `rot`, explicit/implicit padding, fp32 `feat4` | 48 | 768 KiB / 0.750 MiB | Fixed 4-channel feature scaffold; no age or `cov_id` |
| CUDA rasterizer transient inputs, `R=64` | fp32 xy, fp32 scale, fp32 rot, fp32 feat | 276 | 4416 KiB / 4.313 MiB | Plus transient fp32 conic and tile pair buffers |

## Acceptance Decision

Use 32 bytes/Gaussian only as a design target for a not-yet-implemented packed
runtime record at low rank (`R=4` or `R=8`). Do not use it to justify current v6
canvas L2 residency, and do not use it for `R=64`.

For H004 scheduling arithmetic:

- `R=4` or `R=8` packed fp16 canvas state is comfortably L2-resident on both
  GPUs at 16K Gaussians.
- `R=64` packed fp16 canvas state alone still fits both L2 caches, but consumes
  37.5% of 3080 Ti L2 before any rasterizer working set.
- Current fp32 v6 `R=64` canvas plus ST-score state is about 4.56 MiB, consuming
  76% of 3080 Ti L2. Treat that as not L2-resident in practice for the full
  render pipeline.
- 4070 mobile's 32 MiB L2 has enough headroom for all listed canvas-state
  variants; the bottleneck there is more likely bandwidth/occupancy and the
  rasterizer's pair-list/output working set than raw canvas-state residency.

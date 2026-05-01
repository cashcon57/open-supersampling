# OSS-Gaussian Metal Port — Design Notes

**Sprint:** 7 / Track M
**Target hardware:** Apple M3 Max MacBook Pro (16-inch, 38-core GPU, 128 GB unified)
**Tier:** Lite — 5K Gaussian budget @ 1440p
**Scaffold:** `oss/gaussian/ports/metal/`

## Compute kernel design

The MSL port mirrors the vendored Image-GS CUDA tile rasterizer one-for-one. A
threadgroup covers a 16×16 pixel tile (`local_size = 16×16 = 256` threads), one
thread per output pixel. Each threadgroup walks its tile's CSR-encoded Gaussian
list, cooperatively loads candidate Gaussians into `threadgroup` memory in
batches of TOPK, evaluates the 2D quadratic form per pixel, accumulates
weighted features in thread-local FP32 registers, then writes the final
top-K-normalized result to a single output buffer slot.

Subgroup operations move into MSL via `simd_ballot()` / `simd_shuffle()`. The
port must use the runtime `simdgroup_size` rather than hardcoding — Apple's
M-series uses **32-thread simdgroups**, not the 64-thread waves of RDNA. Each
256-thread tile spans 8 simdgroups; each simdgroup independently issues a
ballot to early-skip Gaussians whose bounding box doesn't intersect the lanes'
pixel coordinates.

## Tile size choice

Fixed at 16. This is dictated by `oss.gaussian.renderer.TILE_SIZE` and the
network's `tile_proj` stride. Apple TileMatrix Multiply (TMM) prefers 8×8
fragments, but our kernel is bandwidth-bound on Gaussian fetch, not compute-
bound on accumulation, so 16×16 stays optimal: it amortizes the per-tile
list-load cost over 256 pixels rather than 64.

## CoreML constraints

`coremltools` 8.x converts `Conv2d`, `ConvTranspose2d`, `GroupNorm`, and `SiLU`
natively in the `mlprogram` format. The Sprint 4 network uses exactly these
ops, so no operator workarounds are needed. We pin
`minimum_deployment_target=macOS14` to expose the post-WWDC23 `mlprogram`
runtime, which gives us GPU + ANE co-execution. Default
`compute_units=ALL` lets CoreML pick per-layer placement; in practice the
encoder/decoder convs land on the ANE and the head's small 1×1 conv lands on
the GPU.

CoreML defaults to FP16 weights on GPU/ANE. Acceptance tolerance for the
T7.M.3 parity check is set to 1e-2 mean-abs-diff vs PyTorch FP32 — tighter
would chase quantization noise.

## M3 Max bandwidth scaling

The renderer is memory-bandwidth-bound on Gaussian record fetches:
~32 bytes/Gaussian × N Gaussians per frame, plus the output write.

| GPU | Bandwidth | Predicted relative frame time at same Gaussian count |
| --- | --- | --- |
| RTX 3080 Ti | ~912 GB/s | 1.0× (reference) |
| M3 Max 38-core | ~400 GB/s | ~2.3× |
| Steam Deck APU | ~88 GB/s (LPDDR5) | ~10.4× |

Sprint 1 measured 8K Gaussians @ 1440p in the 1–3 ms range on the 3080 Ti.
Scaling to the M3 Max **at 5K** Gaussians (Lite tier budget) gives an
expected frame time of `1–3 ms × (5/8) × 2.3 ≈ 1.4–4.3 ms` — well within a
60 fps render budget but **not** the headroom needed for a real-time game
integration on top. Sprint 7 ships the offline benchmark only; in-game
integration on macOS is post-v1.

## CrossOver coexistence

T7.M.5 is a smoke test, not a ship-blocking integration. The intent is to
prove that running the native Metal renderer alongside DXMT (CrossOver's D3D-
to-Metal translation layer) doesn't deadlock the GPU. Apple's Metal scheduler
multiplexes command queues across processes; we expect contention to show up
as frame-time variance, not as crashes. Any real game integration on
CrossOver targets v1.1 once Sprint 8+ defines a frame-interception path that
respects Wine's process model.

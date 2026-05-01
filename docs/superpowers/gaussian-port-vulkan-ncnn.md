# OSS-Gaussian Vulkan + ncnn Port — Design Notes

**Sprint:** 7 / Track V
**Target hardware:** Steam Deck (LCD or OLED) — RDNA 2 / Van Gogh APU, 8 CUs, 16 GB LPDDR5
**Tier:** Pico — 1K Gaussian budget @ 1280×800
**Scaffold:** `oss/gaussian/ports/vulkan_ncnn/`

## Compute structure

The GLSL compute port mirrors the CUDA tile rasterizer. Workgroup is
`local_size = 16×16 = 256` threads, one thread per output pixel. Per-tile
shared memory holds up to TOPK Gaussian records loaded cooperatively from
the CSR-encoded `tile_index` / `tile_starts` buffers. Each thread evaluates
the 2D quadratic form, accumulates weighted features in an FP32 register,
and writes the final top-K-normalized result to the storage image.

Subgroup primitives (`subgroupBallot`, `subgroupShuffle`) require
`GL_KHR_shader_subgroup_ballot` + `_shuffle`, both core on RDNA 2 / Mesa
24+. RDNA 2 wave size is **64**, not 32 — the kernel uses `gl_SubgroupSize`
rather than hardcoding. Each 256-thread workgroup spans 4 waves, each
issuing independent ballots for Gaussian bbox-overlap skip.

We enable `VK_KHR_shader_float16_int8` on device creation so Gaussian
records can be stored as packed FP16 (saves ~50% bandwidth on RDNA 2's
LPDDR5), with the per-pixel accumulator staying FP32 to avoid drift on the
1K-Gaussian Pico budget where each Gaussian carries a relatively large
fraction of the final pixel value.

## ncnn op coverage

Audited against the Sprint 4 `GaussianParamNetwork`:

| PyTorch op | ncnn op | Supported via PNNX | Notes |
| --- | --- | --- | --- |
| `Conv2d` (3×3, stride 1/2) | `Convolution` | yes | first-class, FP16 packed on Vulkan |
| `ConvTranspose2d` (`UpBlock`) | `Deconvolution` | yes | first-class |
| `GroupNorm` | `GroupNorm` | yes (ncnn ≥ 20230223) | requires PNNX ≥ 20240410 |
| `SiLU` (`x * sigmoid(x)`) | `Swish` | yes | exact alias |
| `cat` (skip connections) | `Concat` | yes | first-class |
| `Conv2d` 1×1 head | `Convolution` | yes | first-class |

Net result: zero `Custom` layers expected in the exported `.param`. T7.V.3
verifies this by grepping for `Custom` after export and failing CI if any
appear.

## Steam Deck thermal budget

Deck APU is rated for a ~9 W sustained GPU power envelope inside its 15 W
TDP cap. At 60 fps target the per-frame thermal budget is roughly:

- Game render: ~6.5 ms (varies wildly by game)
- gamescope composite + scaling: ~0.5 ms
- **Our Pico-tier render budget: ≤ 4 ms** (T7.V.5 acceptance criterion)
- Display present + scheduler slack: remainder

This leaves the renderer competing with the network inference for the
4 ms budget. ncnn on Vulkan with FP16 packed math hits ~1.5–2 ms for our
Pico-tier checkpoint (~250K params, U-Net at 360×640) on RDNA 2, leaving
~2 ms for the Vulkan rasterizer. At 1K Gaussians the rasterizer is fetch-
bound on ~32 KB of Gaussian records per frame — comfortably under 1 ms on
Deck's 88 GB/s bandwidth.

## Bandwidth-scaling sanity check

Same arithmetic as the Metal port doc:

- 3080 Ti reference: 8K Gaussians @ 1440p ≈ 1–3 ms (Sprint 1 § T1.6).
- Deck Pico: 1K Gaussians @ 1280×800 ≈ `(1–3) × (1/8) × (912/88) ≈ 1.3–3.9 ms`.

p99 frame time at the Pico budget must come in under 4 ms (T7.V.5
acceptance). If it doesn't, the contingency is to drop K_per_tile from 3 to
2, or reduce Gaussian count to 512 — both are runtime knobs, no retraining
required.

## What this port doesn't do (Sprint 7 scope cap)

- No DLL swap. No game integration. The Sintel offline benchmark is the
  only validation path this sprint.
- No gamescope plugin. The legacy `2026-04-30-v0.2-deck-first-design.md`
  explored that path for the pixel-based track; the Gaussian track defers
  it to a post-v1 sprint.
- No on-device training. The Pico checkpoint is exported from a Sprint 4
  training run on Lambda H100 and shipped read-only to the Deck.

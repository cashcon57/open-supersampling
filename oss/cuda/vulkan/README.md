# OSS Vulkan compute shader path (cross-vendor + Steam Deck)

**Status:** scaffolding — no working shaders yet
**Target hardware:** anything Vulkan 1.3+ capable (Steam Deck APU, integrated graphics, mid-range AMD/Intel/NVIDIA)
**Why this matters:** primary path for Steam Deck (RDNA2 APU), secondary path for AMD/Intel where vendor-specific kernels haven't shipped, fallback for the No-ML variant on shader-only GPUs.

## Strategy

The OSS rasterizer + canvas warp + tile bin pipeline can be expressed entirely in Vulkan compute shaders without ML extensions. This is the "no-ML variant" baseline that ships alongside v6.2.

Pipeline (compute shaders only):

1. `validity_mask.comp` — fused MV + depth + material → 1-bit per-tile mask
2. `canvas_warp.comp` — Jacobian-free for translation-dominant Gaussians; fallback path for deformation
3. `tile_bin.comp` — counting sort (3-pass: count, prefix sum, write IDs)
4. `rasterizer_lowrank.comp` — R=4 latent splat with conic row recurrence
5. `composite.comp` — reproject base + canvas residual + tonemap → swapchain

For Steam Deck (RDNA2, no native ML), we use covariance LUT instead of learned spawner; static sub-pixel offsets via blue-noise texture; bilinear/bicubic resolve in fragment shader.

## Build (preview)

```bash
# Compile shaders to SPIR-V
glslangValidator -V oss/cuda/vulkan/shaders/rasterizer_lowrank.comp -o build/rasterizer_lowrank.spv

# Or use Slang for cross-target compilation
slangc oss/cuda/vulkan/shaders/rasterizer_lowrank.slang -o build/rasterizer_lowrank.spv -profile glsl_450
```

## Slang vs GLSL

Slang (NVIDIA-led but cross-vendor) is the recommended source language. It compiles to SPIR-V (Vulkan), HLSL (DX12), and CUDA. Single source = three runtimes.

```hlsl
// rasterizer_lowrank.slang (sketch)
import math;

[shader("compute")]
[numthreads(16, 16, 1)]
void rasterize(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<Gaussian> canvas,
    StructuredBuffer<uint> tile_lists,
    StructuredBuffer<uint> tile_offsets,
    RWStructuredBuffer<float4> output  // R=4 latent
) {
    // ... R=4 latent splat with conic row recurrence
    //     w_{x+1} = w_x * r_x identity for free expf elimination
}
```

## What works

Nothing yet. This is a scaffold.

## What's blocked

- Slang toolchain integration with our build (currently CUDA-only)
- Validation harness for "Vulkan output matches CUDA output bit-for-bit"
- Steam Deck test rig

## What we need

- Steam Deck for testing (or remote dev access)
- Reviewer with Vulkan compute shader experience
- Slang vs raw GLSL decision (recommend Slang per NVIDIA RTX Neural Shaders precedent)

## References

- [Slang language docs](https://github.com/shader-slang/slang)
- [Vulkan compute tutorials](https://www.khronos.org/blog/vulkan-merges-rfcs)
- [Steam Deck GPU spec](https://www.steamdeck.com/en/tech) (RDNA2 APU, Vulkan 1.3 capable)

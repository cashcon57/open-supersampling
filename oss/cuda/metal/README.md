# OSS Apple Metal path (M3+ Tensor Cores)

**Status:** scaffolding — no working kernels yet
**Target hardware:** Apple M3, M4, M5 with hardware Tensor Cores
**Dependencies:** Xcode 15+, MPS (Metal Performance Shaders), MSL (Metal Shading Language)

## Strategy

Apple Silicon has unique tradeoffs:
- Unified memory (no PCIe transfer cost)
- Strong integrated GPU but lower raw FLOPS than discrete cards
- Metal Performance Shaders (MPS) provides matmul / convolution primitives
- Tensor Cores on M3+ (older M1/M2 don't have them)

Pipeline:

1. **Phase 1 — MSL kernel ports** of `rasterizer_fwd.cu` and `rasterizer_bwd.cu` to MSL syntax
2. **Phase 2 — MPSGraph integration** for matmul-heavy paths (TC-GS pattern)
3. **Phase 3 — Core ML export** of the student backbone for M3+ Neural Engine inference (16-core ANE = 38 TOPS)

## Apple-specific opportunities

- **Unified memory**: canvas state can live in shared memory accessible to both CPU and GPU without copy. Different L2-residency story than discrete GPUs.
- **Neural Engine for student backbone**: ANE is great for small CNN inference; could handle ~0.4M student model at very low power. Different runtime path than the Metal compute shaders for the rasterizer.
- **MetalFX**: Apple's own upscaler. We're not competing directly; OSS targets cross-vendor including non-Apple. But MetalFX comparison is the natural Apple-side benchmark.

## What works

Nothing yet. This is a scaffold.

## What's blocked

- No M3/M4/M5 dev rig on hand
- MSL ↔ CUDA kernel translation requires manual rewriting (no automated path)
- Core ML export of distilled student is a separate engineering effort

## What we need

- Apple Silicon dev hardware (M3+ Mac)
- Reviewer with Metal / MPS / Core ML experience
- Decision: ship via Steam (Mac builds) or via standalone framework

# OSS HIP / ROCm path (AMD)

**Status:** scaffolding — no working kernels yet
**Target hardware:** AMD RDNA3+ (RX 7600+, MI300X) with wave64 + matrix cores
**Dependencies:** ROCm 6.0+, hipify-perl, hipcc

## Strategy

HIP is largely source-compatible with CUDA. The fastest path to AMD parity:

1. **Phase 1 — `hipify-perl` translation**: run `hipify-perl` over `oss/cuda/src/*.cu` to generate HIP equivalents in `oss/cuda/hip/src/`. Most kernels translate 1:1; manual fixes only for CUDA-specific intrinsics.
2. **Phase 2 — RDNA3 matrix-core acceleration**: replace `wmma::*` calls with HIP equivalents (`__builtin_amdgcn_wmma_*`) for the rasterizer's W·G matmul.
3. **Phase 3 — wave64 optimization**: AMD wavefront is 64 threads (vs NVIDIA's 32-thread warp). Audit warp-level primitives (`__shfl_*`, cooperative groups) for wave64 correctness.
4. **Phase 4 — LDS-resident canvas**: AMD calls shared memory "LDS"; same role as CUDA shared. Verify L2-residency strategy holds on RDNA3 (Infinity Cache may complicate the analysis).

## Build

```bash
# Translate (one-shot, regenerate when CUDA src changes)
hipify-perl oss/cuda/src/rasterizer_fwd.cu > oss/cuda/hip/src/rasterizer_fwd.hip
hipify-perl oss/cuda/src/rasterizer_bwd.cu > oss/cuda/hip/src/rasterizer_bwd.hip

# Build (requires HIP toolkit)
hipcc -O3 --offload-arch=gfx1100 oss/cuda/hip/src/*.hip -o build/oss_hip.so
```

## What works

Nothing yet. This directory is a scaffold for the AMD kernel port.

## What's blocked

- No AMD hardware on hand for development. Need an MI300X / 7900 XTX loaner or remote access.
- HIP wmma intrinsics differ from CUDA's WMMA API; manual port required for tensor-core paths.
- ROCm support on consumer cards (RX 7600, 7800 XT) is good as of 2026 but driver maturity varies.

## What we need

- AMD hardware access (loan or cloud)
- Reviewer with RDNA3 ISA experience
- Validation suite that confirms HIP outputs match CUDA bit-for-bit (within fp16 atol=1e-3)

# OSS Intel oneAPI / SYCL path (Intel Arc + Xe-HPG)

**Status:** scaffolding — no working kernels yet
**Target hardware:** Intel Arc (A380, A580, A750, A770), Battlemage (B580+)
**Dependencies:** Intel oneAPI 2024+, DPC++/SYCL compiler, Intel GPU drivers

## Strategy

Intel Arc has XMX Matrix Engines (analogous to NVIDIA TC / AMD WMMA) and Xe-cores. SYCL is the cross-vendor C++ heterogeneous compute standard; oneAPI is Intel's flagship implementation.

Pipeline:

1. **Phase 1 — SYCL port** of CUDA kernels using DPC++ (Intel's SYCL flavor). Most CUDA → SYCL translations are mechanical via Intel's `dpct` (DPC++ Compatibility Tool).
2. **Phase 2 — XMX matrix-engine acceleration** via `joint_matrix_*` SYCL intrinsics for the W·G splat matmul.
3. **Phase 3 — ESIMD path** for predicated rasterization (Intel's explicit SIMD extension; can outperform plain SYCL for branchy code).

## Intel-specific opportunities

- **XeSS comparison**: Intel's own upscaler. OSS positions cross-vendor including Intel; OSS benchmarks vs XeSS will be the natural Intel-side measurement.
- **Battlemage and beyond**: Intel's GPU pipeline is gaining momentum. Vendor-neutral SR with first-class Intel support is a competitive advantage.
- **OpenVINO export**: Intel's inference runtime; equivalent role to TensorRT. Student backbone could export to OpenVINO for Intel Arc paths.

## What works

Nothing yet. This is a scaffold.

## What's blocked

- No Intel Arc dev card on hand
- SYCL kernel translation needs manual fixes for matrix-engine intrinsics
- OpenVINO export of distilled student is a separate engineering track

## What we need

- Intel Arc A770 or B580 dev hardware
- Reviewer with oneAPI / SYCL / OpenVINO experience
- Validation that XMX matrix-engine outputs match CUDA TC bit-for-bit (within fp16 tol)

## References

- [Intel oneAPI SYCL programming guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)
- [`dpct` CUDA-to-SYCL translation tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html)
- [Intel ESIMD extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)

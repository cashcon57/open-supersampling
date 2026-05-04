# Vendor Optimization Audit Reference

**Purpose:** Sprint 6 performance-pass reference for OSS-SR temporal inference. This note pins the vendor ground truth that should replace transcript-memory claims when designing custom kernels.

**Scope:** Matrix-acceleration paths relevant to fusing the v5-pixel-temporal forward pass into vendor-specific inference kernels. This is not a benchmark result.

## Summary Matrix

| Vendor / target | Matrix path | Practical precision floor | Cooperative execution | OSS implication |
|---|---|---|---|---|
| NVIDIA Ampere/Ada/Hopper | Tensor Cores via CUDA WMMA, PTX `mma.sync`, CUTLASS/CuTe; Hopper adds WGMMA/TMA paths | FP16/BF16/TF32/INT8; newer architectures add FP8/FP4 families | Warp-level for WMMA/`mma.sync`; warpgroup-level for Hopper WGMMA | Primary S6 target. RTX 3080 Ti (Ampere, SM86) should use FP16 `mma.sync.aligned.m16n8k16`-class conv tiles where layout permits. |
| AMD CDNA / Instinct | MFMA / Matrix Core instructions; rocWMMA on supported CDNA architectures | FP16/BF16/INT8 and newer FP8/MX formats depending on generation | CDNA rocWMMA targets wave64 | Good data-center/HPC path, not Steam Deck. Treat as a separate HIP backend from RDNA desktop. |
| AMD RDNA 3+ desktop | WMMA instructions via ROCm/rocWMMA support for RDNA architectures | Primarily FP16/BF16/INT8-style packed math depending on exact GPU/toolchain | rocWMMA lists RDNA architectures as wave32 | Plausible desktop AMD path for RDNA 3+. Do not assume RDNA 2 has this path. |
| AMD RDNA 2 / Steam Deck | No documented matrix accelerator path comparable to RDNA 3 WMMA or CDNA MFMA | Vector ALU / packed dot-product fallback only | Wave32/64 shader execution, no tensor/matrix core equivalent | Steam Deck ceiling is materially lower. Treat Vulkan/HIP compute fallback as the shipping path; expect roughly `~25%` of matrix-equipped vendor peak until measured otherwise. |
| Intel Arc / Xe | XMX via DPAS, exposed through oneAPI/SYCL and lower-level tooling | FP16/BF16/INT8 depending on device and compiler path | Subgroup / systolic DPAS execution | Use OpenVINO/oneAPI first; custom XMX kernels are a later specialist backend. |
| Apple Silicon | MPS/Metal SIMD-scoped matrix multiply where available; BNNS/Accelerate and Core ML/ANE for framework path | FP16/BF16 where exposed by framework/hardware; exact custom-shader matrix support varies by GPU family | Metal SIMD-group features, not CUDA-like warps | Prefer Core ML / MPSGraph / BNNS first. Custom Metal compute needs device-family gating from Metal Feature Set Tables. ANE is not generally programmable as a custom shader target. |

## Notes By Vendor

### NVIDIA

Ground truth:

- CUDA exposes Tensor Cores through the C++ WMMA API and lower-level PTX `mma` / `mma.sync` instructions.
- CUTLASS is NVIDIA's supported template path for high-performance GEMM/conv kernels and explicitly spans Volta through Blackwell, mixed precision, narrow integer, and newer low-bit formats.
- CUTLASS/CuTe is the best starting point for a fused inference prototype because it already models threadblock tiles, warp tiles, layouts, and epilogue fusion.

S6 guidance:

- Target Ampere first because the live training/deploy box is RTX 3080 Ti.
- Use FP16 weights/activations with FP16 or FP32 accumulation only after a quality check; the current production path already favors TRT FP16 over INT8 on Ampere.
- Do not design around one monolithic 1.25 MB shared-memory resident model. The full 626K-param FP16 model is ~1.25 MB, but Ampere per-block shared memory is far smaller. Tile weights into shared memory per layer/block and keep only the active convolution tile resident.
- Candidate MMA tile for conv-im2col lowering: `mma.sync.aligned.m16n8k16` for FP16 on SM80/SM86, but validate register pressure and layout overhead against CUTLASS kernels.

Sources:

- NVIDIA PTX ISA, `mma`/`mma.sync`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA CUDA C++ Programming Guide, WMMA/Tensor Cores: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUTLASS docs: https://docs.nvidia.com/cutlass/latest/overview.html

### AMD

Ground truth:

- CDNA/Instinct is the MFMA/Matrix Core family. It is the AMD path closest to NVIDIA Tensor Core programming.
- RDNA 3 introduced WMMA-style matrix instructions; RDNA 2 should not be treated as matrix-equipped for OSS purposes.
- rocWMMA differentiates supported CDNA wave64 targets and RDNA wave32 targets.

S6 guidance:

- Split AMD into two backends:
  - **HIP MFMA/rocWMMA** for CDNA and RDNA 3+ where matrix instructions are available.
  - **Vulkan/HIP vector fallback** for RDNA 2 / Steam Deck.
- Be explicit in docs and UX: Steam Deck is not a matrix-accelerator target. It may run OSS via a lite/distilled model and hand-tuned vector shader path, or fall back to FSR 2.
- Any "Wave64 is always better" claim from transcript memory should be discarded. Wave width is architecture/toolchain specific; RDNA 3 rocWMMA support is wave32, CDNA support is wave64.

Sources:

- AMD RDNA 3 ISA reference: https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
- AMD CDNA / Instinct ISA docs: https://www.amd.com/en/search/documentation/hub.html
- ROCm rocWMMA API reference: https://rocm.docs.amd.com/projects/rocWMMA/
- ROCm Matrix Core programming note: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/

### Intel

Ground truth:

- Intel XMX executes DPAS operations on 2D systolic arrays.
- The practical application path is oneAPI/SYCL, OpenVINO, or lower-level DPAS intrinsics after the model path is stable.

S6 guidance:

- Use OpenVINO first for validation.
- A custom XMX backend only makes sense after CUDA/HIP prove that fused small-conv inference is worth the engineering cost.

Sources:

- Intel XMX optimization guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/latest/xmx.html
- Level Zero specification: https://spec.oneapi.io/level-zero/latest/

### Apple

Ground truth:

- Apple exposes optimized ML/matrix paths through Accelerate/BNNS, Metal Performance Shaders, MPSGraph, Core ML, and device-family-gated Metal shader features.
- ANE is best reached through Core ML; it is not a generic custom compute-shader target.
- Metal Feature Set Tables now include device-family gating for SIMD-scoped matrix multiply. Do not assume every Apple Silicon Mac/iOS device has the same custom Metal matrix path.

S6 guidance:

- Start with Core ML / MPSGraph export and measure.
- If custom Metal is needed, gate by Metal Feature Set Tables and keep a BNNS/MPS fallback.
- Avoid claiming tile-based deferred rendering helps compute kernels. TBDR matters to render pass architecture; compute performance must be measured through Metal counters.

Sources:

- Apple Accelerate / BNNS: https://developer.apple.com/accelerate/
- Apple MPSMatrixMultiplication: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- Apple Metal Feature Set Tables: https://developer.apple.com/metal/capabilities/
- Apple ML research note on ANE / MPS backend: https://machinelearning.apple.com/research/neural-engine-transformers

## Open Questions

- What is the smallest v5-pixel-temporal tier that preserves the v4 LPIPS gain once temporal state is active?
- Does fusing the temporal head with the backbone epilogue reduce latency enough to beat TRT FP16 by `>=3x`, or is the bottleneck memory bandwidth / pixel shuffle layout?
- Can the bicubic/bilinear skip path be fused without wasting MMA occupancy on simple interpolation?
- Which exact RDNA 3 desktop GPUs expose rocWMMA cleanly enough for shipping kernels through HIP?
- Is Intel Arc custom XMX worth maintaining, or should OpenVINO own that path?
- Which Apple GPU families expose SIMD-scoped matrix multiply and enough threadgroup memory for a useful custom Metal conv tile?

# CUDA Mega-Kernel Design Memo

**Purpose:** Sprint 6 design sketch for a custom NVIDIA path that fuses the v5-pixel-temporal forward pass into fewer CUDA dispatches.

**Status:** Design note only. No benchmark result. Use this to decide what to prototype after Sprint 5 quality is locked.

## Target

Target workload:

- Input: 1080p LR + G-buffer stack, 12 channels.
- Output: 4K RGB.
- Model: v5-pixel-temporal standard tier, currently about 626K parameters.
- Live GPU target: RTX 3080 Ti (Ampere / SM86).
- Current deploy baseline: PyTorch/ONNX/TensorRT FP16 path.

Forward pieces to fuse where practical:

1. SR backbone convolutions.
2. Pixel-shuffle / upsample path.
3. Previous-HR warp input consumption.
4. Disocclusion-gated temporal head.
5. Final residual/skip add and clamp.

## Why Fuse

The model is small enough that kernel launch overhead and global-memory round trips can dominate layer math. A custom path should aim to:

- keep tiles of activations in registers/shared memory across adjacent layers;
- fuse activation functions and epilogues into producer kernels;
- avoid writing full-resolution intermediate tensors to global memory;
- use Tensor Core MMA for convolution tiles when channel dimensions align.

CUTLASS describes the useful hierarchy for this: threadblock tiles, warp tiles, instruction tiles, and epilogue fusion. The target is not "one literal CUDA block computes the whole network." It is fewer larger persistent/fused kernels with weight/activation tiles staged close to compute.

## Weight Residency Reality Check

626K parameters at FP16 is about 1.25 MB. That does **not** fit in one Ampere threadblock's shared memory. The practical design is tiled:

- Stage one layer's active filter tile into shared memory.
- Use `cp.async`-style global-to-shared pipelining where available.
- Keep only the current input activation tile and a narrow halo in shared memory.
- Write global memory only at necessary resolution/phase boundaries.

Approximate shared-memory budgeting:

- Weight tile: target `<= 32-48 KB`.
- Activation tile + halo: target `<= 32-48 KB`.
- Double buffering: only if occupancy remains acceptable.
- Leave budget for warp-level reductions, mask/debug output, and bank-conflict padding.

## MMA Shape Sketch

Ampere FP16 Tensor Core target:

- PTX instruction family: `mma.sync.aligned.m16n8k16.*.f16.f16.*`.
- Conv lowering: implicit-GEMM tile over `(output_pixels x output_channels)` by `(kernel_extent * input_channels)`.
- Preferred layers: channel-rich backbone/head convs.
- Less suitable layers: PixelShuffle and simple skip interpolation, which are layout/memory movement operations rather than matrix math.

Initial mapping:

| Layer class | Candidate implementation |
|---|---|
| 3x3 conv with enough channels | implicit-GEMM Tensor Core tile |
| 1x1 conv / projection | direct GEMM tile |
| ReLU / sigmoid / scalar gates | fused epilogue |
| PixelShuffle | custom vectorized store/reindex, likely separate from MMA tile |
| Warp previous HR | texture-like bilinear sample; probably separate fused prepass unless temporal head can consume warped tile directly |
| Bicubic/bilinear skip | vector interpolation path; fuse final add if output tile is already resident |

## Proposed Prototype Order

1. **Measure current TRT layer breakdown** at the target input shape.
2. **Fuse temporal head only** first. It runs at HR and may be launch/memory heavy.
3. **Fuse backbone conv blocks** using CUTLASS implicit GEMM templates or a small custom `mma.sync` microkernel.
4. **Pull PixelShuffle into the producer epilogue** only after confirming layout costs.
5. **Fuse warp + head consumption** if profiling shows the warped previous HR write/read is costly.

Do not start with a fully hand-written whole-model kernel. Build a measurable prototype around the top 2-3 kernels from the TRT profile.

## Open Questions

- Does the standard tier's channel shape align well enough for Tensor Core tiles, or should S6 distillation choose channel counts that are Tensor-Core friendly?
- Is the temporal head the true latency bottleneck at 4K, or is the backbone/pixel-shuffle path still dominant?
- Can the disocclusion mask be computed tile-local without materializing a full HR mask tensor?
- Where should the first-frame bilinear `prev_hr` initialization live in the export/runtime path?
- Does CUDA Graph replay plus TRT already remove enough launch overhead that custom fusion must focus on global-memory traffic instead?
- How much quality is lost if the skip path is bilinear-only and fused, matching the existing ONNX production compromise?

## References

- NVIDIA PTX ISA, `mma.sync`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA CUDA C++ Programming Guide, WMMA/Tensor Cores/shared memory: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUTLASS documentation: https://docs.nvidia.com/cutlass/latest/overview.html
- NVIDIA CUTLASS blog on GEMM hierarchy and epilogue fusion: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
- FlashAttention-2 CUTLASS/Hopper fused-kernel case study: https://arxiv.org/abs/2312.11918
- NVIDIA Tensor Core programmability overview: https://arxiv.org/abs/1803.04014

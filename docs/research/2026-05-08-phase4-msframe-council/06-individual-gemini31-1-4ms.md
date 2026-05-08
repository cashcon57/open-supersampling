# Gemini 3.1 Pro Thinking — Individual Response (1–4ms Budget Re-assessment)

Source: model council, 2026-05-08. Verbatim individual response (not synthesized).

## The Setup (Hardware target: ASUS G14 = RTX 4070 Mobile, 8GB)

**[OSS-team note: Gemini benchmarks against 4070 mobile = the OSS dev rig. ~256-336 GB/s bandwidth, ~22 TFLOPS FP32, ~110-130 TFLOPS BF16 TC. Tighter than 4070 desktop. Numbers below are correct for 4070 mobile, not desktop.]**

In 2.5 ms, can only move ~750 MB across VRAM bus. PyTorch kernel launch overhead alone can consume 1ms.

To match DLSS 4 / Frame Gen budgets, must radically pivot. Gaussian canvas can no longer be a heavy latent space; it must become a dumb, ultra-fast temporal reservoir, and the neural network must become microscopic and strictly localized.

## The 1-4ms Reality Check: What Dies Today

- **HAT-Tiny must die (for inference).** Even smallest windowed transformer eats 3-6ms at 1080p due to memory round-trips. Need a 3-layer hardware-fused CNN (NAFNet-micro / custom TensorRT graph) running FP16/INT8.
- **Cross-Attention must die.** SDPA too slow when invoked per-pixel/per-window. Replace with Raster-Fusion: rasterizer spits canvas features directly onto pixel grid, just Concat with LR.
- **F=64 must die.** Cannot accumulate 64 channels. Drop to R=4 (RGB + 1 confidence).
- **PyTorch eager mode must die.** Entire inference path captured in single CUDA Graph or exported to TensorRT/ONNX.

## The 3.0 ms "Nano" Architecture Budget (RTX 4070 Mobile)

### Phase 1: G-Buffer Masking & Canvas Warp (0.3 ms)
**Goal:** Move canvas, figure out what pixels need neural updates.

- Gaussians store **only 32 bytes**: xy (FP16×2), conic (FP16×3), rgb (FP16×3), confidence (FP16). 16k Gaussians = 512 KB. **Entire canvas fits in GPU L2 cache.**
- Warp kernel: read MV, apply `xy' = xy + MV(xy)`. Skip Jacobian unless `∇·V > 0.1`.
- Active Masking: single pass comparing depth to historical depth. Output 1-bit mask per 16×16 tile.

### Phase 2: O(N) Tile Binning (0.3 ms)
**Goal:** Assign 16k Gaussians to screen tiles without `torch.sort`.

Custom 2-pass histogram (counting) sort:
1. Atomic add → count Gaussians per tile
2. Prefix sum (exclusive scan) → array offsets
3. Write Gaussian IDs into pre-allocated offsets

N=16k, Tiles ≈ 8k → counting sort virtually instantaneous.

### Phase 3: The R=4 Rasterizer (1.0 ms)
**Goal:** Render canvas to target resolution instantly.

- Rasterize only 4 channels: RGB and Confidence (sum of weights)
- Row-Recurrence Conic: `w_{x+1} = w_x * r_x`. No `expf` per pixel
- 2×2 register tiling (4 pixels/thread) to maximize math-to-memory ratio
- Output: 4-channel FP16 tensor at target resolution (`Canvas_Out`)

### Phase 4: Sparse Neural Resolve & Spawner (1.0 ms)
**Goal:** Fix artifacts and handle disocclusions, only where necessary.

Block-Sparse:
- Read 1-bit tile mask from Phase 1. Typically 10-20% of tiles "active"
- Tiny TensorRT-compiled CNN (3×3 depthwise, 1×1 pointwise, GELU, 3 layers) on active tiles only
- Input: `Concat(Upsampled_LR, Canvas_Out_RGB, Canvas_Confidence)`
- Output: `RGB_Residual`

Spawner: NO dense MLP. If pixel has low Canvas_Confidence AND marked disoccluded → hard-spawn at exact pixel center, inheriting RGB from LR.

### Phase 5: Final Composite & TAA/FXAA (0.4 ms)
`Final_RGB = Canvas_Out_RGB + RGB_Residual`. Standard lightweight spatial AA if edge variance high.

## Step-by-Step Execution Plan

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | R=4 factorization: rasterizer outputs F=4, pass into `nn.Conv2d(4+3, 3, 1)` | 136ms → ~5-15ms (register spilling + L2 thrashing cease) |
| 2 | Conic row-recurrence: `w_{x+1} = w_x * r_x` in CUDA inner loop | ~5-15ms → ~1-3ms |
| 3 | Custom binning: rip out `torch.sort`, use CUB `DeviceRadixSort` or custom counting sort | Saves ~5ms framework/sorting overhead |
| 4 | Sparsity & CUDA Graphs: active tile mask + `torch.cuda.make_graphed_callables` or `cudaGraphLaunch` | Eliminates Python overhead, locks latency to GPU compute |
| 5 | Distillation: freeze HAT-Tiny v6.1/v6.2 → train 3-layer MobileNet block or single NAFNet block to mimic on active tiles → export TensorRT FP16 | Replaces backbone |

## Why This Beats DLSS 4 (Conceptually)

DLSS uses monolithic deep neural network to **hallucinate** detail. We use explicit, advectable geometric cache (Gaussians) to **bypass** the need for massive network. If Gaussians represent 90% of image perfectly via pure analytical warping, neural network only needs to be 10% the size of DLSS for same result.

## Key Math Insights

### Canvas state size
```
xy: FP16×2 = 4 bytes
conic: FP16×3 = 6 bytes (a, b, d)
rgb: FP16×3 = 6 bytes
confidence: FP16×1 = 2 bytes
Total minimum: 18 bytes
Aligned to cache line (32 bytes) for fastest access

16k Gaussians × 32 bytes = 512 KB
RTX 4070 L2 = 36 MB (overkill); RTX 4070 Mobile L2 ~32 MB
→ Canvas fits in L2 with massive headroom
```

### Why Phase 1+2 fits in 0.6ms on 4070 Mobile
- Warp: 16k Gaussians × ~10 FLOPs = 160 KFLOPs (compute-trivial)
- Bandwidth: 16k × 32 bytes = 512 KB read, 512 KB write
- At 256 GB/s, 1 MB transfer = 4μs → orders of magnitude under budget

### Why R=4 raster fits in 1ms
- Per pixel: K_tile (~8) Gaussians × 4 channels × 2 ops (mul+add) = 64 FLOPs
- 1080p output: 2M pixels × 64 = 128 MFLOPs
- At 110 TFLOPS BF16 TC: 128 MFLOPs / 110 TFLOPS = 1.2μs (compute-trivial)
- BUT: bandwidth-bound on Gaussian state read. K=8 × 32 bytes × 2M pixels access pattern is the actual bottleneck.

## Architecture Pivot Statement

The Gaussian canvas is not the upscaler. The Gaussian canvas is an **L2-resident temporal detail cache** that **analytically warps for free**, **corrects only where reprojection fails**, and **never runs a heavy neural network in the critical display path**.

[OSS-team note: 32-byte struct claim is implementation choice (alignment), not fundamental. L2 residency is design-friendly but not guaranteed in shipping context due to engine compute contention.]

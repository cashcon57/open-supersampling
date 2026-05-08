# Claude Opus 4.7 Thinking — Individual Response (1–4ms Budget Re-assessment)

Source: model council, 2026-05-08. Verbatim individual response (not synthesized).

## Re-Assessment: 1–4ms Total Frame Budget (DLSS4-class)

Prior synthesis targeted ~5-7ms on a 4070. That's already 1.5-2× over budget for DLSS4 parity. DLSS4 Transformer on Ada runs ~1.5-2.5ms at 1440p→4K, DLSS-FG adds ~1-2ms. To hit 1-4ms total for upscale + extrapolation, structural cuts needed, not optimizations.

## Brutal Truth About 1–4ms Budget on RTX 4070

At 4ms on RTX 4070 (~29 TFLOPS FP32, ~120 TFLOPS BF16 TC):
- ~480 GFLOPs Tensor Core budget (BF16)
- ~115 GFLOPs FP32 budget
- ~1.2 GB HBM bandwidth (at 288 GB/s × 4ms)

HAT-Tiny (~54.9G MACs) at 50% util consumes ~2-3ms alone — cannot live in budget.

**[OSS-team note: this arithmetic is contested. At full BF16 TC throughput (120 TFLOPS), 54.9G MACs = ~0.46ms not 2-3ms. Opus's "50% util at FP32" is pessimistic; HAT-Tiny is TC-eligible. Verify with measurement before committing to ≤0.4M student.]**

STSS reference: 4.4ms/frame at 1080p with 0.4M parameters for combined SR + extrapolation. Architectural North Star.

## What Must Be Cut Entirely

| Component | Reason | Replacement |
|-----------|--------|-------------|
| HAT-Tiny as runtime backbone | 9M params, ~54.9G MACs | Distilled student: ~0.4M params (NAFNet-nano / 3-block EfficientViT-lite) |
| 64-channel feature rasterizer | 64-channel splat dominates bandwidth | R=4 latent, direct-to-RGB residual |
| Global pixel↔Gaussian cross-attention | 200-500μs overhead at 2000 windows | Rasterized fusion: Σw_g·z_g/Σw_g + 1×1 conv |
| `expf` per Gaussian-pixel | Transcendentals dominate on Ada | Row-recurrence + covariance LUT hybrid |
| `torch.sort` in tile binning | 8-12ms alone | Radix bin + persistent tile lists |
| Separate forward/backward at inference | Doubles memory + compute | Inference graph has no backward at all |
| FP32 throughout kernel boundary | Bandwidth-bound | FP8/BF16 canvas state; FP16 accumulate; FP32 only for guards |

## Revised Budget Allocation (1080p, RTX 4070, 4ms total)

| Stage | Budget | Mechanism |
|-------|--------|-----------|
| Preprocess + disocclusion mask (fused) | 0.15 ms | Single dispatch: LR + MV + depth → packed residual + validity bitmap |
| Temporal warp (Jacobian-free, persistent conic) | 0.20 ms | Streaming copy for rigid (~90%); branch to JΣJ^T only on ∇·V > ε |
| Tile rebin (incremental, persistent lists) | 0.10 ms | Only Gaussians crossing boundaries (~5-10%) |
| Low-rank rasterizer (R=4, TC-accelerated, recurrence) | 1.40 ms | W·G mapped to mma.sync BF16→FP32; LUT path for narrow |
| Tiny student backbone on active tiles | 0.80 ms | INT8 QAT, ~0.4M params, residual-gated to ≤30% LR pixels |
| Rasterized fusion + 1×1 decode | 0.35 ms | No attention; G(p) = Σw_g·z_g/(ε+Σw_g), 3-layer pointwise |
| Budgeted spawner (Kalman + births only) | 0.25 ms | Births capped 256/frame; Kalman 6 FLOPs/Gaussian |
| HR composite + tonemap → swapchain | 0.15 ms | DX12/Vulkan interop, no readback |
| Graph launch + sync | 0.10 ms | Single CUDA Graph replay |
| **Total** | **~3.50 ms** | Headroom for 4K via residual path |

For DLSS4+FG parity, add ~1ms for extrapolated frame (essentially free in architecture — second rasterizer ~1.4ms minus shared setup). Net: ~4.5ms for upscale + 1 generated frame.

## Reordered Priority Stack (Budget-Constrained)

### P0 — Non-Negotiable Architectural Cuts

1. **Kill HAT-Tiny from inference path.** Train student with HAT as teacher, target ≤1M params, ≤5G MACs at 1080p LR. EfficientViT multi-scale linear attention: up to 6.4× speedup over Restormer for SR. STSS achieves SOTA in 0.4M params / 4.4ms.

2. **R=4 latent channel rasterizer.** F=64 dead at this budget. `f_g = B·z_g`, z_g ∈ ℝ⁴. Rasterizer emits RGB + 4 latent; post-raster 1×1 decodes. ~16× less accumulation work.

3. **Replace cross-attention with rasterized fusion.** Reserve local top-K (K=16, 1-head) only for tiles flagged by disocclusion mask — typically <5% of windows.

### P1 — Tensor Core Conversion

4. **Map splat accumulation to mma.sync / wmma.** TC-GS (May 2025): alpha-blend→matmul mapping yields 2.18× over prior 3DGS accelerators, 5.6× cumulative. Sum-composite EWA has no ordering dependency — cleaner mapping than 3DGS.

5. **Pad all dims to TC multiples.** head_dim=30 is a latent perf bug. Pad to 32 immediately.

6. **INT8 QAT student backbone.** TensorRT for RTX: 1.5× over DirectML in Unreal at 1080p (5.7ms→3.8ms on 5090). For ≤1M param student, INT8 PTQ typically <0.1dB PSNR.

### P2 — Overhead Elimination

7. **CUDA Graph capture of entire frame DAG.** ~50 launches × 5μs = 250μs overhead. Capture once, replay forever. NNE's RDG async path is engine-integration equivalent.

8. **Persistent tile lists with incremental updates.** ~5-10% of Gaussians cross tiles per frame. Maintain sparse delta lists; full rebin only on scene cuts.

9. **Radix bin replacing torch.sort.** Bounded tile count → O(N) counting sort, ~1ms → ~50μs.

### P3 — Arithmetic & Bandwidth Reduction

10. **Conic row-recurrence.** `w_{x+1} = w_x · r_x`, `r_{x+1} = r_x · exp(−a)`, `Δ²q_x = 2a`. Reduces expf from per-pixel to ~2/row. On Ada, transcendentals run ~1/4 FMA rate — direct wall-clock win.

11. **Covariance codebook + LUT for narrow Gaussians.** Quantize to 16-64 prototypes × phase-quantized offsets. Branch: LUT for σ<4px, recurrence for medium, separable for large.

12. **Jacobian-free warp branch.** If |∇·V| < ε → Σ' = Σ + Δt·D (pure translation). ~90% of Gaussians; ~8-12 FLOPs saved each.

13. **Quad+subgroup hybrid gradient reduction (training).** 10× backward speedup. Reduces atomicAdd to ~5%.

### P4 — Temporal Amortization

14. **Multi-rate execution.** Student backbone at 30Hz; rasterizer + warp at display rate (120Hz). Disocclusion-only cross-attention at 30Hz. FSR3 pattern.

15. **Validity mask + stream compaction.** 60-85% pixels reproject-stable at 60fps. Compact remainder into 1D active array; full raster only there.

16. **Frame extrapolation is structurally free.** Deterministic canvas warp IS the frame generator. Second warp+raster pass at +Δt/2 costs only rasterizer (~1.4ms). Beats DLSS-FG's optical flow accelerator dependency.

## Where This Departs From Prior 5-7ms Plan

| Prior | Revised | Reason |
|-------|---------|--------|
| Distill HAT to EfficientViT-lite (P4/P5) | Moved to P0; ≤1M params | At 4ms, even distilled HAT-class too large |
| R=8 latent target | R=4 baseline; R=8 quality toggle | Rasterizer is critical path; halving rank ~2× |
| Local top-K K=32-64 | K=16 only on disocclusion tiles | Any global attn violates budget |
| 5-7ms target on 4070 | 3.5ms; 4.5ms with FG | DLSS4 parity = DLSS4 budget |
| INT8/FP8 optional | Mandatory for student | Bandwidth precludes FP16 runtime |
| Half-res splat + analytic gradient | Quality-mode only | Added decoder complexity; R=4 full-res simpler |
| Backbone at half temporal rate | 30Hz backbone, 120Hz raster (4:1) | Aggressive amortization mandatory for FG-class |

## Fallback Tiers

| Tier | Trigger | Response |
|------|---------|----------|
| 0 (normal) | <3.5ms | Full pipeline |
| 1 | 3.5-4.0ms | R=4→R=2; disable spawner 1 frame |
| 2 | 4.0-4.5ms | Skip backbone; warp+raster only |
| 3 | >4.5ms | Bicubic + canvas composite; flag for next-frame budget |

Canvas capacity dynamic: `S = √(residual_energy × motion_entropy)`. Hallway: ~4k. Foliage: ~12k.

## Validation Targets (v6.2)

- 1080p → 4K upscale: 3.0-3.5ms on 4070 (DLSS4 Performance ~2.5ms)
- Upscale + 1 extrapolated: 4.0-4.5ms (DLSS4 + FG on Ada ~3.5-5ms)
- Quality floor: PSNR within 0.3dB of HAT-teacher; LPIPS within 0.02
- VRAM: <400MB for model + canvas at 4K

The architectural insight: frame extrapolation is a side-effect of deterministic canvas advection, not a separately-trained head. That structural property lets you hit DLSS4+FG budgets with a single network — not paying for two models.

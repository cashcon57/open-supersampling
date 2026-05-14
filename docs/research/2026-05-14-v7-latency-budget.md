# v7 latency budget — from CPU bench to <2 ms Pico target

**Date:** 2026-05-14
**Source data:** `scripts/bench_v7_inference.py` output (CPU, see also `2026-05-14-v7-inference-bench.md`).
**Bar to clear:** STSS reports **4.35 ms @ 1080p, 0.4M params, PSNR 35 / LPIPS 0.018** on RTX 3090 fp16. OSS Heavy student must match; OSS Pico student needs <2 ms.

## Current state (Python reference rasterizer, CPU)

| HR shape | Variant | Total ms/frame | bottleneck |
|---|---|---|---|
| 480×640 (TartanAir) | baseline | 648 | rasterizer |
| 480×640 | pc+mip | 696 | rasterizer |
| 1080×1920 | baseline | **4370** | rasterizer |
| 1080×1920 | pc+mip | **4595** | rasterizer |
| 2160×3840 | — | (extrapolated) ~17000 | rasterizer |

Time scales ~linearly with `canvas_count × HR_area`. Mip-Splatting filters add ~5-8%; parent-child adds ~3%. The rasterizer pure Python loop is the dominant cost (~95% of total).

## Projection: 3080 Ti GPU (CUDA native rasterizer)

CPU→3080 Ti speedup factor for this workload: **~30-50×** based on prior v6.x rasterizer ports.

| HR shape | est ms/frame on 3080 Ti | STSS-comparable? |
|---|---|---|
| 480×640 | ~22 | training-only |
| 1080×1920 | **~146** | ❌ 30× over STSS's 4.35 |
| 2160×3840 | ~570 | ❌ 130× over |

**Conclusion:** the Python-reference rasterizer is sufficient for training (where wall-time per step is ~5–10 s and bottlenecked by backward, not forward), **but insufficient for deployment inference**. The Phase 4 CUDA / Triton port is on the critical path for a shippable model.

## Per-component allocation, target

Working backward from STSS's 4.35 ms @ 1080p budget, here's where v7 Pico student has to fit each component:

| Component | OSS-Pico budget @ 1080p | What it does |
|---|---|---|
| LR stem (CNN, ~50K params) | **0.3 ms** | 9-ch LR → F=64 feature map at LR resolution |
| Canvas refresh (every 8 frames, amortized) | **0.5 ms** | Teacher's spawn + canvas-update path runs sparsely; amortized per-frame cost is teacher_cost / 8 |
| Canvas render at t_now | **0.6 ms** | The CUDA rasterizer's job; 32K Gaussians at 1080p |
| ERM stack (4 blocks × ~16K params) | **0.4 ms** | Local 5×5 ReLU-linear attention fuses LR + HR-canvas |
| Head (3×3 conv → RGB) | **0.1 ms** | F → 3 channels |
| Bicubic anchor + skip | **0.05 ms** | F.interpolate |
| **Total** | **~1.95 ms** | |

Total ~0.4M params: 50K stem + ~64K × 4 ERM + ~3K head = **~310K params** (plus the teacher canvas which is separate state). Leaves ~90K param budget for any extra residual blocks.

## What has to change to get there

### Critical (none of these are optional)

1. **CUDA / Triton rasterizer kernel.** The single biggest item. Target: ~30× faster than Python at 1080p / 30K Gaussians. Phase 4. Reference: gsplat's CUDA kernel handles the same primitive at ~0.5 ms / 100K Gaussians on a 3090.

2. **fp16 / bf16 throughout.** STSS reports fp16; we should match. Halves memory + roughly halves compute for matrix ops.

3. **Static canvas allocation.** The current `NDCanvasState.add()` grows the canvas at every spawn. For inference, pre-allocate the maximum capacity once at warmup; no runtime growth. (Training already pre-allocates; just need to confirm the inference path doesn't re-allocate.)

### Highly recommended

4. **Teacher rate-limit.** Run the heavy HAT-Tiny backbone every 4–8 frames, not every frame. The student handles in-between via canvas time-extrapolation through V_xt. Amortizes the ~10–20 ms teacher cost across multiple frames.

5. **Mip-Splatting filter fusion into the rasterizer.** Currently `_apply_3d_smoothing_filter` and `_apply_2d_mip_filter` are separate Python passes that compute det() per Gaussian. In the CUDA kernel, fold them into the existing Gaussian projection step — adds the `+ s · I` term and the opacity rescale at near-zero extra cost.

6. **AABB culling via LSH.** Already discussed in the v7 spec (Phase 4). At 30K Gaussians the current naive 3-sigma cull becomes the kernel hot path. LSH at 3D (where AABB starts failing due to the t-axis) recovers ~2× perf at no quality loss.

### Probably needed

7. **ClassSR-style adaptive compute.** Easy patches (sky, flat walls, motion-blurred regions) go through a fast bilinear-only path; hard patches (edges, text, foliage) use the full ERM stack. ~30% time savings on average game content per STSS's similar mechanism.

8. **Quantization-aware training.** fp8 / int8 for embedded targets (Switch 2, mobile). Standard PTQ should work; QAT if metrics regress > 0.2 dB.

## Sanity check: where does v7 Pico beat STSS structurally?

STSS uses a U-Net + cross-attention. They have ~0.4M params and run at 4.35 ms. We have:

- Same param budget target (~0.4M)
- Same input format (LR + buffers; with NoV+stencil expansion to match)
- **Different state representation**: STSS uses a per-frame fixed-shape feature cache; we use a persistent N-D Gaussian canvas
- **Different temporal mechanism**: STSS uses warp + correction; we use V_xt-encoded native time
- **Different cross-attention**: STSS uses 5×5 local ReLU-linear; we'll match (see ERM block spec)

The architectural delta is the persistent Gaussian canvas. If our V_xt time encoding holds up, we get frame extrapolation at no extra cost — STSS pays for it separately. That's the OSS-FX bet.

## What this means for Phase 3 + Phase 4

- **Phase 3 (pico-005 teacher training):** No latency concern. The 22 ms / step CPU time × 30 GPU speedup → ~700 µs forward per sample on GPU, which is fine for training even at B=2.
- **Phase 4 (CUDA kernel + Pico student distillation):** **THE** gating item for a deployable model. Without the CUDA rasterizer port, OSS can't ship a real-time upscaler regardless of how good the teacher is. Schedule budget: 2–3 weeks for the rasterizer port + 2 weeks for the ERM-style student + 1 week for distillation tuning.
- **Phase 5 (cross-engine fine-tune):** Latency stays the same; this phase is just about data domain shift. Falls out of Phase 4.

## Decision points for the user

1. **CUDA kernel: write it ourselves or fork gsplat?** gsplat handles 3D Gaussian splatting but assumes a perspective camera projection. Our 2D+t time-slice math is different (Schur complement marginalization, not perspective projection). Forking gsplat saves 60% of the work but adds maintenance burden. Writing from scratch is cleaner but longer.

2. **ERM block window size 3 vs 5?** STSS chose 5. With our persistent canvas providing strong spatial prior, we may be able to drop to 3. Difference: ~15% inference speed. Ablation has to come after the student trains.

3. **Quantization aggressiveness.** fp16 is safe and matches STSS. fp8/int8 are aggressive and may regress quality. Pico tier targets matter: if we need Switch-2-class hardware, fp8 is non-negotiable.

These decisions don't need to be made today — they're for Phase 4 kickoff. Listed here so they're not forgotten.

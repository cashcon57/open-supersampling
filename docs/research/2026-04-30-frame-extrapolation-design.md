# Frame Extrapolation Design — ORS ORU-FX

**Status:** Design / pre-implementation  
**Date:** 2026-04-30  
**Scope:** α-conditioned real-time frame extrapolation for arbitrary framerate upscaling

---

## Goal

Add a frame extrapolation module (ORU-FX) to ORS that generates only the frames needed to hit a user's target framerate. Handles 60→90fps, 60→120fps, and similar non-2× boosts natively. No BFI — dropped from scope.

**Primary architecture: G-buffer-assisted extrapolation via DLL hook.** A proxy `dxgi.dll` / Vulkan layer intercepts `Present()` and extracts the depth buffer and motion vectors the game already computed for its own TAA/DLSS/FSR pass — zero extra render cost. These are fed to ORU-FX alongside the color frame to produce a full-res extrapolated frame at t+α. The user only ever sees full-res frames.

**Why not color-only like GFFE?** Intel deliberately went color-only for driver-level portability (no hook required) and as a research claim ("G-buffer free is competitive"). For ORS via DLL hook, depth + motion vectors are available for free on any modern game with TAA, and they eliminate the heuristic estimation GFFE has to do from color alone. On older games without TAA, fall back to color-only gracefully.

**Why not the 1/4-res guide frame?** Requires injecting an extra render pass into the game engine — needs SDK cooperation or very deep render pipeline hooks. Deferred to V2. The depth buffer from the DLL hook gives most of the same positional benefit at zero cost.

---

## Reference Architecture: Intel GFFE

**Paper:** "G-Buffer Free Frame Extrapolation" (Bálint et al., ACM ToG Dec 2024, arXiv:2406.18551)

Key insight: extrapolation (no future frame needed) is feasible with a heuristic warp + lightweight neural correction network.

### ORU-FX pipeline

```
DLL hook / Vulkan layer intercepts Present():
  color(t)          ──►  History Tracker  ──►  Motion history
  depth(t)          ──►  Geometry-aware warp  ──►  Warped_est
  motion_vec(t-1→t) ──►  Flow extrapolation  ──►  F_{t→t+α}
  Warped_est + history + depth + alpha_embed
                    ──►  SCN (neural net)  ──►  Frame_t+α (full-res, displayed)
```

Inputs extracted at zero extra render cost — the game already computed depth and motion vectors for TAA. On games without TAA (no motion vectors available), fall back to estimating flow from color frames only.

1. **Flow extrapolation** — motion_vec(t-1→t) linearly extrapolated: F_{t→t+α} = α × motion_vec. No separate flow estimator needed when motion vectors are present (saves ~3ms vs RAFT-Small).
2. **Geometry-aware warp** — uses actual depth(t) to detect depth discontinuities and prevent color bleeding across edges. Replaces GFFE's luminance Laplacian heuristic with ground-truth depth.
3. **History Tracker** — accumulates temporal feature maps from prior frames for disocclusion fill and motion continuity.
4. **SCN** — takes warped estimate + depth + history features + α embedding, outputs residual correction. Handles disocclusion fill, shading correction, sub-pixel motion.

The SCN is the only learned component.

### GFFE latency (measured, Table 4 of paper)

| Resolution | Total | SCN only |
|-----------|-------|----------|
| 540p      | 2.34ms | 0.90ms |
| 720p      | 3.66ms | 1.30ms |
| 1080p     | 6.62ms | 2.30ms |

Hardware: RTX 4070 Ti Super. Competing methods (UPR-Net=43ms, DMVFN=20.57ms, IFR-Net=19.50ms) are 3–6× slower and require future frames.

**Target for ORS:** ≤8ms at 1080p on Steam Deck GPU (RDNA2, roughly half of RTX 4070 Ti Super → expect ~13ms at 1080p, ~7ms at 720p). 720p @ 60→90fps is the primary Steam Deck use case.

---

## α-Conditioned Architecture

Instead of always generating the midpoint frame (t+0.5), we condition the network on a temporal offset α ∈ (0, 1] representing how far ahead to predict relative to the render interval.

### Why α-conditioning

- **Arbitrary target fps**: to go from 60→90fps, we need frames at t+0.33 and t+0.67 (α=0.33 and α=0.67 relative to the 1/60s render interval). Not 0.5.
- **Fewer frames generated**: 60→75fps needs only α=0.2 and α=0.4 and α=0.6 and α=0.8 ... actually for 60→75fps we need 1 inserted frame every 4 rendered frames (75/60 = 1.25×). Only α=0.2 is needed.
- **Quality gradient**: small α (close to last rendered frame) → easier prediction, higher quality. Large α (close to next rendered frame) → harder, potentially lower quality. Users can tune.

### α schedule math

Given rendered fps `F_r` and target display fps `F_d`:
- If `F_d / F_r` is not an integer, we insert `ceil(F_d/F_r) - 1` frames between rendered frames
- Frame insertion positions (in units of render intervals): `k / ceil(F_d/F_r)` for k=1...(ceil-1)

Examples:
| Render fps | Target fps | Frames per render interval | α values |
|-----------|-----------|--------------------------|----------|
| 60 | 90 | 2 (1 inserted) | 0.5 |
| 60 | 120 | 3 (2 inserted) | 0.33, 0.67 |
| 60 | 75 | ~1.25 (1 inserted every 4th) | 0.2, 0.4, 0.6, 0.8 (alternating) |
| 30 | 60 | 2 (1 inserted) | 0.5 |

The scheduler decides which α to use for each display vsync based on accumulated phase.

### SCN α-conditioning

α is embedded as a sinusoidal positional encoding and concatenated to the SCN bottleneck feature:

```python
def alpha_embed(alpha: float, dim: int = 32) -> Tensor:
    # Standard sinusoidal embedding, dim=32 sufficient for a scalar
    i = torch.arange(dim // 2, dtype=torch.float32)
    freq = 1.0 / (10000 ** (2 * i / dim))
    x = alpha * freq
    return torch.cat([x.sin(), x.cos()], dim=-1)  # (dim,)
```

The SCN is trained with α sampled uniformly from [0.1, 0.95] per sample, forcing it to learn the full temporal interpolation space, not just α=0.5.

---

## ORU-FX Module Design

### Scope boundaries

**IN scope:**
- Heuristic warp pipeline (BG collection, history tracking, geometry-aware warp)
- SCN neural network with α-conditioning
- α scheduler (computes which frame to generate at each display vsync)
- Training pipeline (Sintel + Vimeo-90K, multi-α batches)
- Integration with ORU-Pico output (FX takes Pico's upscaled frames as input)
- ONNX export for deployment
- Vulkan/NCNN inference path (same as ORU-Pico via `vulkan` extra)

**OUT of scope:**

- 1/4-res guide frame render — deferred to V2, requires render pipeline injection
- BFI (Black Frame Insertion) — dropped
- Interpolation (needs future frame) — extrapolation only
- Per-game tuning profiles (v1 ships one universal model)
- HDR10 / HDR400 path (deferred to v2)

### Module layout

```
ors/
  model/
    oru_fx.py            # SCN architecture + alpha embedding
    oru_fx_warp.py       # Heuristic warp pipeline (BG collector, history, geometry warp)
  train/
    train_fx.py          # Training script
    losses_fx.py         # Perceptual loss for extrapolation (L1 + SSIM + temporal smooth)
  data/
    sintel_fx.py         # Sintel dataset adapter for extrapolation training
    vimeo90k_fx.py       # Vimeo-90K adapter
  export/
    export_fx.py         # ONNX export
```

### SCN architecture

Small U-Net variant, ~2M params (larger than Pico because spatial coherence matters more for extrapolation artifacts):

```
Input: [warped_t+α (3ch), depth_t (1ch), history_feat (C_h ch), alpha_embed (32)] → concat on channel dim
  ↓ EncoderBlock 32ch
  ↓ EncoderBlock 64ch
  ↓ Bottleneck 128ch + alpha_embed injected here
  ↑ DecoderBlock 64ch
  ↑ DecoderBlock 32ch
Output: residual (3ch) → warped_t+α + residual = Frame_t+α
```

When motion vectors are unavailable (no TAA), flow is estimated from color frames via a lightweight estimator (adds ~3ms). depth_t channel is zeroed and the network falls back to luminance-based edge detection learned during training.

Residual formulation keeps the warp as a strong prior; SCN only corrects shading errors and disocclusion fill.

### Warp pipeline (oru_fx_warp.py)

**Primary path (motion vectors available):**

1. **Flow extrapolation** — game motion vectors linearly extrapolated: F_{t→t+α} = α × motion_vec(t-1→t). No separate flow estimator.
2. **Depth-aware warp** — warp color(t) using F_{t→t+α}, masking pixels that cross depth discontinuities in depth(t) (∆depth > threshold → disoccluded).
3. **Background collection** — pixels with |motion_vec| < threshold across last N frames → BG pool for disocclusion fill.
4. **Disocclusion fill** — masked pixels filled from BG pool.

**Fallback path (no motion vectors — color + depth only):**

1. **Flow estimation** — RAFT-Small (pretrained, frozen) estimates F_{t-1→t} from color frames. Adds ~3ms.
2. **Depth-aware warp** — same as primary path using estimated flow.
3. **Background collection + disocclusion fill** — same as primary.

**Color-only fallback (no motion vectors, no depth):**

1. RAFT-Small flow estimation.
2. Luminance Laplacian as depth discontinuity proxy (GFFE-equivalent).
3. Background collection + disocclusion fill.

---

## Training Data

### Sintel (primary)

- MPI Sintel dataset: 23 scenes × 1024 frames, 1080p, with ground-truth optical flow
- Use clean pass (no motion blur) for training, final pass for validation
- Generate extrapolation targets by holding out every 3rd frame as pseudo-GT, train to predict it from frames t-2 and t-1

### Vimeo-90K (secondary)

- 89,800 7-frame sequences from Vimeo, 448×256
- Same holdout strategy: predict frame 4 from frames 1-3 at various α values
- Rich real-world motion diversity (natural video, not game-like)

### Augmentation

- Horizontal flip
- Random temporal reversal (t+k → predict t+k-1, equivalent to reverse α)
- α jitter: sample α uniformly from [0.1, 0.95] per training example
- Color jitter (hue ±0.05, saturation ±0.1) — keep small, color accuracy matters

### NOT using

- NoiseBase — denoising dataset, not appropriate for extrapolation
- Game captures — licensing unclear, defer to v2 with in-house captures

---

## Integration with ORU-Pico

FX sits downstream of Pico in the rendering pipeline:

```
Raw LR frame + G-buffers → ORU-Pico (denoising + SR) → HR clean frame → ORU-FX → display
```

FX operates on HR output. Pico runs at render fps (e.g., 60fps native on Steam Deck). FX runs at display fps (e.g., 90fps). For every rendered frame, FX may need to generate 0 or 1 extrapolated frame depending on the α schedule.

This means FX must run in ~11ms budget at 90fps display (1000ms/90 = 11.1ms), leaving the remainder for Pico. At 720p, GFFE target is ~3.66ms — comfortably fits.

FX is optional and independently deployable: users on higher-end hardware can use Pico without FX; users wanting higher display fps enable FX.

---

## Performance Targets

| Device | Resolution | Render fps | Target display fps | FX budget | Feasibility |
|--------|-----------|-----------|-------------------|-----------|-------------|
| Steam Deck | 720p | 40 | 60 | 16.7ms | ✓ easy |
| Steam Deck | 720p | 60 | 90 | 11.1ms | ✓ (expect ~7ms) |
| Steam Deck | 800p | 60 | 90 | 11.1ms | marginal |
| Mid-range PC | 1080p | 60 | 120 | 8.3ms | ✓ (expect ~8ms) |
| RTX 4070 Ti | 1080p | 120 | 144 | 6.9ms | ✓ (6.62ms measured) |

---

## Metrics

**Quality** (higher is better):
- PSNR on Sintel test set (targets: >30dB for α=0.5, >27dB for α=0.9)
- SSIM on Sintel test set (target: >0.90)
- LPIPS on Vimeo-90K test set (target: <0.10)

**Temporal stability**:
- Warp error = mean pixel displacement between consecutive extrapolated frames vs GT (target: <2px at 720p)
- Flicker metric = temporal variance on static regions (target: <0.5% of peak brightness)

**Latency**:
- End-to-end (warp + SCN inference) at 720p on Steam Deck APU (target: ≤8ms)
- SCN-only at 720p (target: ≤3ms)

---

## Implementation Timeline (estimate)

| Week | Milestone |
|------|-----------|
| 1-2 | Heuristic warp pipeline (BG collector, history, flow extrapolation) |
| 3-4 | SCN architecture + α-conditioning + single-GPU smoke test on Sintel |
| 5-6 | Training on Vimeo-90K, PSNR/SSIM baseline |
| 7-8 | ONNX export + NCNN/Vulkan inference path |
| 9-10 | Latency profiling on Steam Deck, tuning for budget |
| 11-12 | Integration with ORU-Pico output, end-to-end test |

**MVP:** weeks 1-8 (quality model with ONNX export). Weeks 9-12 are deployment/integration.

Total: ~10-12 weeks to deployable MVP alongside Pico.

---

## Open Questions

1. **RAFT-Small vs lighter flow estimator**: RAFT-Small is ~10ms on GPU; for very tight budgets, we may need to train a dedicated optical flow head into the SCN. Revisit at week 9 profiling.
2. **BG pool size**: how many frames to accumulate for background. GFFE uses a fixed ring buffer (paper doesn't specify size). Start with 8 frames, tune.
3. **Phase-misaligned vsync**: when display vsync does not align neatly with α schedule (e.g., variable refresh rate), need a continuous α prediction path. Defer to v2.
4. **Multi-frame extrapolation (α > 1.0)**: generating two frames ahead. GFFE does not support this; quality degrades rapidly. Not planned.

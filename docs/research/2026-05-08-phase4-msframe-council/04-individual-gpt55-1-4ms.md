# GPT-5.5 Thinking — Individual Response (1–4ms Budget Re-assessment)

Source: model council, 2026-05-08. Verbatim individual response (not synthesized).

## Reassessment: 1–4 ms Total Budget Means a Different Product Architecture

Hard answer: the previous optimized architecture is still too expensive. A 5–7 ms target is plausible for an experimental Gaussian upscaler; a DLSS4 / DLSS4+FG-class 1–4 ms target requires turning the Gaussian canvas into a sparse temporal correction layer, not the full upscaler.

DLSS4 introduced transformer Super Resolution and Multi Frame Generation; DLSS4.5 uses FP8 / Tensor Core acceleration to keep heavier models tractable. DLSS4 frame generation replaced the older per-generated-frame optical-flow + AI path because that was cost-prohibitive. Standard: no big per-frame dynamic graph, no global attention, no dense high-channel raster, no full scene reprocessing.

For Gaussian-canvas approach, new rule:

> Every displayed frame gets only reprojection + sparse correction + tiny resolve. Heavy neural update is asynchronous, amortized, and optional.

## 0. New Budget Reality

| Mode | Target | Contents |
|------|--------|----------|
| Ultra-fast SR only | ~1.0 ms | Reprojection, validity mask, tiny sparse Gaussian sharpen, simple resolve |
| Quality SR | ~2.0 ms | Reprojection, canvas warp, low-rank sparse raster, tiny INT8/FP8 resolve |
| SR + 1 generated frame | ~3–4 ms total | Source SR ~2–2.5 ms + generated extrapolation ~0.8–1.5 ms |

Current 136 ms forward at N=4096, 540×960, F=64 is the wrong hot-path contract — needs ~50–100× effective reduction.

## 1. What Must Be Removed From Hot Path

| Component | Verdict | Replacement |
|-----------|---------|-------------|
| HAT-Tiny every frame | Too expensive | Teacher only; runtime tiny student or no neural update |
| Global pixel↔Gaussian cross-attn | Impossible at 1–4ms | Raster-fusion + optional local attention on active tiles only |
| 64-channel HR Gaussian raster | Dead | R=4–8 low-rank latent splat |
| Dense 16k canvas raster every frame | Dead | Active tile + K-per-tile capped sparse raster |
| `torch.sort(tile_id,gid)` | Dead | Persistent tile bins / CUDA counting sort |
| Dense spawner regression | Dead | Disocclusion-only birth + Kalman-style correction |
| Per-pixel `expf` over every overlap | Dead | LUT or row recurrence |
| Backward/autograd-style state path | Training only | Inference state = conic + low-rank + confidence |

Correct product runtime is closer to TAAU + neural resolve + sparse Gaussian detail injection.

## 2. Revised Product Pipeline

### Displayed source frame path
```
Input: LR color, motion vectors, depth, optional normals/roughness/material,
       previous HR output, persistent Gaussian canvas

Hot path:
  1. motion/depth validity mask
  2. reproject previous HR
  3. warp Gaussian canvas
  4. sparse low-rank Gaussian correction on invalid/high-detail tiles
  5. tiny resolve/composite shader
```

### Generated frame path
```
Input: latest source-frame output, current motion vectors, warped canvas

Hot path:
  1. late frame reprojection
  2. canvas extrapolation
  3. sparse correction only for holes/edges
  4. resolve/composite
```

Generated frame must NOT run HAT, cross-attention, or spawner.

## 3. Target Runtime Budget Table (1080p)

| Stage | 1ms mode | 2ms mode | 4ms SR+FG mode |
|-------|---------|---------|----------------|
| Preprocess MV/depth/validity | 0.08–0.15 | 0.10–0.20 | 0.15–0.25 |
| HR reprojection | 0.15–0.25 | 0.20–0.35 | 0.25–0.45 |
| Canvas warp | 0.03–0.08 | 0.05–0.12 | 0.08–0.15 |
| Sparse Gaussian raster | 0.25–0.45 | 0.45–0.90 | 0.80–1.40 |
| Tiny resolve / sharpen | 0.15–0.25 | 0.25–0.45 | 0.40–0.70 |
| Async neural update amortized | 0 | 0.10–0.30 | 0.30–0.80 |
| Frame-gen extrapolation | — | — | 0.80–1.30 |
| Overhead / graph / composite | 0.10 | 0.15 | 0.25 |
| **Total** | **~0.8–1.3** | **~1.5–2.5** | **~3.0–4.5** |

For laptop 4070 / 8GB target: design around 2ms SR path and ~3.5–4ms SR+FG path.

## 4. Core Algorithm

### 4.1 Validity mask first
```
p' = p − u_t(p)
V(p) = 1[|D_t(p) − D_{t−1}(p')| < τ_D]
     · 1[‖∇u_t(p)‖_F < τ_U]
     · 1[M_t(p) = M_{t−1}(p')]

I_base(p) = I_{t−1}^HR(p')   if V(p)=1
            cheap fallback   if V(p)=0
```

Only invalid / edge / disocclusion pixels eligible for Gaussian work.

### 4.2 Tile priority scheduler
```
P(B) = λ_R · max R(p) + λ_D · max(1−V(p)) + λ_E · max ‖∇I‖ + λ_U · max ‖∇u‖_F
R(p) = ‖I^LR(p) − Down(I_base)(p)‖_1
```

Process top tiles until budget exhausted: `Σ K_B · P_pix(B) · R_latent ≤ W_budget`.

## 5. Low-Rank Raster (Mandatory)

```
f_g ≈ B z_g,  z_g ∈ ℝ^R,  R = 4–8

Z(p) = (Σ w_g(p) z_g) / (ε + Σ w_g(p))
ΔI(p) = φ_θ(Z(p), m(p), I_base(p), D(p), u(p))
I_t^HR(p) = I_base(p) + ΔI(p)
```

| Mode | R | Tile K cap | Support |
|------|---|------------|---------|
| 1ms | 4 | 8 | 2σ |
| 2ms | 6 | 12–16 | 2–2.5σ |
| 4ms | 8 | 16–24 | 2.5σ |

## 6. Replace Cross-Attn With Raster-Fusion

```
G(p) = (Σ w_g z_g) / (ε + Σ w_g)
F'(p) = F(p) + ψ_θ(F(p), G(p), m(p), D(p), u(p))
```

If attention survives at all: only active tiles, only local Gaussians, K≤16, head_dim padded to 32, one batched kernel call, no per-window dispatch.

## 7. Gaussian Weight Eval

### 7.1 LUT codebook (fast path)
```
Σ_g ≈ s_g² · Σ_{k(g)},  k ∈ {1..M},  M = 16 or 32
phase_x, phase_y ∈ {0..7}

L[k, φ_x, φ_y, dy, dx] = exp(−½ Δp^T Σ_k^−1 Δp)

Runtime: w = lut[cov_id][phase_x][phase_y][dy][dx];
```

No conic eval, no expf. Default for small/medium Gaussians.

### 7.2 Row recurrence (exact fallback)
```
q(x,y) = a·dx² + 2b·dx·dy + d·dy²
Δq_x = q(x+1,y) − q(x,y) = a(2dx+1) + 2b·dy
Δ²q_x = 2a   ← constant

w_x = exp(−0.5 q_x)
r_x = exp(−0.5 Δq_x)

w_{x+1} = w_x · r_x
r_{x+1} = r_x · exp(−a)
```

One/two exp per row, not per pixel.

Three kernels:
- **LUT splat** — quantized small/medium Gaussians
- **Row-recurrence splat** — continuous conics / high-quality mode
- **Reprojection-only fallback** — stable / overflow tiles

## 8. Tensor Core Use (Conditional)

```
Y = W·Z,  W ∈ ℝ^{P×K},  Z ∈ ℝ^{K×R},  Y ∈ ℝ^{P×R}
```

For 16×16 tile, P=256. If K≥16 and R∈{8,16}, WMMA useful. If K<12 or R=4, scalar/vector CUDA may be faster (matrix too skinny).

```
if K_tile < 12: scalar LUT/recurrence kernel
else: batched WMMA W·Z kernel
```

Pad all ML channel dims to TC-friendly: head_dim 30→32, latent R 6→8, MLP hidden mod 16/32.

## 9. Spawner Out of Critical Path

### 9.1 Disocclusion-only birth
Only where V(p)=0 AND R(p) > τ_R. Births = TopK(P(B)). Hard limits: max 128–512 new/frame, max 1–4 new/tile.

### 9.2 Kalman correction for existing
```
x̂_{t|t} = x̂_{t|t−1} + K_t (z_t − H x̂_{t|t−1})
K_t = P_t / (P_t + R_t)   (diagonal)
```

Few FLOPs per Gaussian. Spawner MLP for birth, not maintenance.

### 9.3 Checkerboard fix at this budget
Spawn at disoccluded pixel center: `xy_g = p + (0.5, 0.5)`. Then advect: `xy_g' = xy_g + u(p)·Δt`. Motion field naturally moves Gaussian off-grid. If needed, add deterministic blue-noise offset from small texture (not regressed).

## 10. Temporal Warp — Jacobian-Free Default

```
δ(p) = |∇·u(p)|

if δ(p) < ε:  Σ' = Σ + Δt·D   (or Σ' = Σ for stable)
else:         Σ' = JΣJ^T + Δt·D
```

Persist conic Λ = Σ^−1. Fast transport: `Λ' ≈ A^−T Λ A^−1`, A = I + Δt∇u. Only deformation tiles take full path.

## 11. Dynamic Degradation Ladder

Estimate work: `W_frame = Σ P_B · K_B · R`. If `W_frame > W_budget`, degrade in order:

1. Increase validity threshold
2. Reduce K_tile: 24→16→8
3. Reduce R: 8→6→4
4. Reduce support: 2.5σ→2.0σ
5. Skip low-priority disocclusion fill
6. Reuse previous HR tile
7. Reduce neural resolve quality
8. Disable Gaussian correction entirely

DLSS-like frame pacing: quality varies, frame time does not.

## 12. Mode Configurations

### Mode A — 1ms competitive
- No HAT, no cross-attn, no dense spawner, no full raster
- R=4, K_tile≤8, active tiles ≤10–15%
- LUT weights only, tiny linear resolve
- `I_t = Reproject(I_{t−1}) + φ_θ(Z_4, m, D, u)`

### Mode B — 2ms quality
- R=6 or 8, K_tile≤16, active tiles ≤25%
- LUT + recurrence
- Tiny depthwise CNN resolve
- Async low-rate canvas refresh

### Mode C — 3–4ms SR+FG
- Source: ~2.0–2.5 ms
- Generated: ~0.8–1.5 ms
- `I_{t+α} = I_t(p − α·u_t(p)) + M_hole · ΔI_gauss`

## 13. Build Order

| P | Item | Action |
|---|------|--------|
| P0 | Low-rank residual splat | feat64 → z4/z8; emit Z_R + weight_sum + confidence |
| P1 | Validity mask + tile scheduler | Most tiles do not run |
| P2 | LUT Gaussian kernel | cov_id uint8, phase 3-bit, z fp16/int8 |
| P3 | Persistent tile bins | Update only Gaussians crossing boundaries |
| P4 | Tiny resolve shader | 1×1 → depthwise 3×3 → 1×1, INT8/FP8 |
| P5 | Move HAT/cross-attn to teacher/offline | training, distillation, async refresh, screenshots |

## 14. Final Verdict

Displayed-frame path is not a neural Gaussian renderer. It is reprojection-first temporal reconstruction with sparse Gaussian residual correction.

Minimum viable 1–4ms architecture:
```
Persistent state: xy, conic (or cov_id), z4/z8, confidence, age, tile membership

Per displayed source frame:
  validity mask, HR reprojection, canvas warp,
  sparse low-rank Gaussian correction, tiny resolve

Per generated frame:
  late reprojection, canvas extrapolation, hole/edge correction only

Async/amortized: spawner, HAT/student backbone, cross-attn (if any), tile rebalance
```

Council priority shifts: validity-mask reprojection-first becomes P0, low-rank residual splat becomes P0, HAT and global cross-attn leave hot path, spawner becomes disocclusion-only, rasterizer becomes budgeted+sparse+LUT-driven+capped, generated frames never run full SR.

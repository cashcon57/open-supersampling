# H004 — L2-Resident Canvas + Budgeted Tile Scheduler

**Status:** `untested`
**Class:** System-level engineering (composition of known patterns)
**Filed:** 2026-05-08
**Source:** Gemini 3.1 standalone; combined with GPT-5.5's tile-budget formula

## Claim

A persistent 2D Gaussian canvas for SR + frame extrapolation can be made **L2-resident** (no HBM round-trips for canvas state) by packing per-Gaussian state into ~24-32 bytes:

```
Per-Gaussian (FP16 packed):
  xy:         FP16 × 2  = 4 bytes
  conic Λ:    FP16 × 3  = 6 bytes
  rgb or z:   FP16 × 3  = 6 bytes
  confidence: FP16      = 2 bytes
  age, cov_id: UINT8 × 2 = 2 bytes
  pad to alignment       = 24-32 bytes total

16k Gaussians × 24 bytes = 384 KB
RTX 4070 L2 = 36 MB
RTX 4070 mobile L2 ≈ 32 MB
```

**Canvas fits in L2 with massive headroom** on all reference targets.

Combined with a **budgeted tile scheduler**:

```
P(B) = λ_R · max R(p)        (residual energy)
     + λ_D · max (1−V(p))    (disocclusion)
     + λ_E · max ‖∇I‖         (image gradient)
     + λ_U · max ‖∇u‖_F       (motion gradient)

Budget:  Σ_{B ∈ Active} K_B · P_pix(B) · R ≤ W_budget
```

Process tiles in priority order until budget exhausted. **Hard frame-time control.**

## Performance claim

- **L2 residency**: zero HBM reads/writes for canvas state during warp/raster steps in isolation
- **Real-world**: hit-rate dependent on engine context (L2 is shared with engine compute)
- **Budgeted scheduler**: deterministic frame pacing — quality varies, frame time does not (within bounds)

## Quality claim

Quality varies by available budget. At max budget all tiles processed → equivalent to non-budgeted baseline. Below max budget, low-priority tiles fall back to reprojection-only → quality degrades gracefully on low-residual / high-validity regions where degradation is least visible.

## Test plan

1. **Profile L2 hit rate** via Nsight Compute on isolated raster kernel (no other GPU work):
   - Measure L2 read miss rate during warp + raster passes
   - Target: <5% L2 miss for canvas state reads
2. **Profile L2 hit rate inside engine integration** (e.g., simple Unreal scene):
   - Measure same metric with engine compute work running concurrently
   - Determine: is "L2-resident" a guarantee or a target?
3. **Budgeted scheduler ablation**: compare full-canvas raster vs budgeted scheduler at multiple budget levels (50%, 70%, 90% of max). Plot quality (PSNR/LPIPS) vs budget vs frame-time variance.
4. **Worst-case stress**: dense foliage / particle-heavy scenes that maximize active tile priority. Verify scheduler holds frame-time within target.

## Acceptance gate

- L2 hit rate >95% in isolated kernel; >70% in engine context
- Frame-time variance under budgeted scheduler: stddev < 15% of mean across diverse scenes
- Quality at 90% budget: PSNR within 0.3 dB of full budget

## Compose with

- **H002 low-rank splat** — smaller per-Gaussian payload makes L2-fit easier
- **H001 conic recurrence** — L2-resident state means recurrence inputs stay hot in L1/L2

## Risks

- **L2 contention** with engine compute kernels in shipping context. "L2-resident" may be a design target, not a guarantee. Treat as cache-friendly design, not delivered guarantee.
- **Scheduler λ tuning**: priority weights need empirical calibration across scene types
- **Foliage / particle worst-case**: may need fallback to canvas-capacity scaling if scheduler can't maintain budget

## Lab notes

(empty — untested as of 2026-05-08)

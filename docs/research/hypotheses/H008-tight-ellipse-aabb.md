# H008 — Tight Ellipse AABB Culling

**Status:** `validated` — math validated (6/6 unit tests, 2026-05-08); CUDA edit landed in `oss/cuda/src/rasterizer_fwd.cu`
**Class:** Engineering optimization (no quality impact; reduced tile-list length)
**Filed:** 2026-05-08
**Source:** Phase 4 priority stack v4 + GPT-5.5 council

## Claim

Replace the conservative scale-radius AABB `radius = 3 · max(sx, sy)` with the tight per-axis ellipse bound from the conic (a, b, d):

```text
r_x = sqrt(τ · d / (a·d − b²))
r_y = sqrt(τ · a / (a·d − b²))

τ = 9   (3-sigma; matches prior 3·scale convention)
```

Derivation: conic quadratic form `q(x,y) = a·dx² + 2b·dx·dy + d·dy²`; the τ-Mahalanobis ellipse is `q = τ`. Per-axis extent:

- `∂q/∂y = 0 → dy = -b·dx/d`
- Substitute back: `dx² · (a·d − b²)/d = τ` → `r_x = √(τ·d/(a·d−b²))`
- Similarly `r_y = √(τ·a/(a·d−b²))`

For axis-aligned Gaussians (b=0), tight AABB reduces to `(3·sx, 3·sy)` exactly. For rotated anisotropic Gaussians, tight is meaningfully smaller per-axis.

## Performance claim

Reduces tile-list length proportional to AABB area savings. Smaller tile lists → faster radix bin → fewer pixel-Gaussian pair tests.

**Measured:** 1000-Gaussian Monte Carlo (anisotropic, random rotation, scale ∈ [0.5, 3]):
- Average area savings: **34.4%**
- Median area ratio (tight/conservative): **0.666**
- 4:1 anisotropic at 90° rotation: **75% area savings** (worst conservative case)

Translates to ~30% reduction in mean tile-overlap count, less work for the radix bin and the per-tile pair scanner.

## Quality claim

**Zero quality impact.** This is a strict subset of the conservative AABB — every Gaussian still gets binned, just into fewer tiles. The 3σ ellipse is fully contained.

## Test plan — completed

1. ✅ **Math validation in Python (`tests/perf/test_h008_tight_ellipse_aabb.py`, 6/6 pass)**:
   - Axis-aligned matches `3·sx, 3·sy` exactly
   - Isotropic is rotation-invariant
   - 4:1 anisotropic at 90° rotation gets correct r_x=12, r_y=3
   - 3σ ellipse boundary always contained in tight AABB (verified via parametric sweep)
   - AABB corner outside 3σ ellipse (validates we don't crop the support)
   - 1000-trial savings summary: avg 34.4% area reduction
2. ✅ **CUDA implementation** (`oss/cuda/src/rasterizer_fwd.cu` lines 74-93): tight bound applied with degenerate-conic fallback to scale-radius safety bound
3. ⏳ **End-to-end CUDA test**: existing rasterizer test suite must pass (run during commit)
4. ⏳ **Microbench**: tile-bin throughput before/after on real Gaussian batches

## Acceptance gate

- ✅ Math identity tests pass
- ⏳ CUDA test suite passes (within atol=1e-5)
- ⏳ Tile-bin throughput improves measurably (≥10% reduction in pair count is the lower bar; ≥30% target)

## Compose with

- **H001 conic row-recurrence** — both attack the rasterizer hot path, orthogonal axes
- **D8 cull radius** (validated discovery: 3σ is right default) — H008 keeps τ=9 (3σ) as the threshold

## Risks

- **Numerical degeneracy** when `a·d ≈ b²` (singular conic). Fallback path in CUDA handles via `det_inv > 1e-12f` check → reverts to scale-radius bound.
- **FP32 precision** under `sqrtf` could shift edge-tile inclusion by ±1 tile in pathological configs. Validated at fp64 in Python; CUDA fp32 should be within tile boundary.

## Lab notes

### 2026-05-08 — math VALIDATED + CUDA implemented

**Test:** `tests/perf/test_h008_tight_ellipse_aabb.py`

- 6/6 unit tests pass on first run
- Area savings: avg=34.4%, median ratio=0.666, worst-case (4:1 anisotropic, 90° rotation) = 75% savings
- Axis-aligned reduces to `(3·sx, 3·sy)` exactly — backward-compat for the easy case
- Parametric ellipse boundary sweep confirms 3σ isocontour always contained in tight AABB

**CUDA implementation** (`oss/cuda/src/rasterizer_fwd.cu` lines 74-93):

- Compute `det_inv = a*d - b*b`
- If `det_inv > 1e-12f`: tight ellipse formula (τ=9)
- Else: fallback to `3*max(sx, sy)` (existing conservative behavior; preserves correctness for degenerate conics)

**Pending:** end-to-end CUDA test verification + microbench. Edit landed; next commit cycle should run the rasterizer test suite to confirm no regressions.

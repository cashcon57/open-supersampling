# H001 — Conic Row-Recurrence for EWA Gaussian Weights

**Status:** `partial` — math identity VALIDATED at fp64 (Python reference + 8/8 unit tests, 2026-05-08); CUDA port pending
**Class:** Algorithmic identity (no quality loss claimed; pure performance)
**Filed:** 2026-05-08
**Source:** GPT-5.5 individual response (Phase 4 council 1-4ms ask)

## Claim

For an EWA Gaussian with conic `Λ = [a b; b d]` and quadratic form `q(x,y) = a·dx² + 2b·dx·dy + d·dy²`, the per-pixel weight `w(x,y) = exp(−q/2)` along a fixed scanline can be evaluated by recurrence with **constant second difference**.

```text
Δq_x  = q(x+1,y) − q(x,y) = a(2·dx + 1) + 2b·dy
Δ²q_x = 2a   (constant along the row)

w_x = exp(−q_x/2)
r_x = exp(−Δq_x/2)

w_{x+1} = w_x · r_x
r_{x+1} = r_x · exp(−a)
```

**Cost:** 2 exponentials per Gaussian-row, then pure FMAs across pixels.
**vs naïve:** 1 exponential per Gaussian-pixel pair.

## Performance claim

For a 16×16 tile, **256 expf calls per Gaussian-tile → ~16-32 expf calls per Gaussian-tile** (16× transcendental reduction).

On Ada/Ampere, transcendentals run at ~1/4 the rate of FMAs. Direct wall-clock win on rasterizer arithmetic: **2-4× on the hot inner loop**.

## Quality claim

**Bit-exact within float rounding** of naïve formulation. This is a mathematical identity, not an approximation. Any deviation from naïve `expf` is float-precision drift that should be bounded.

## Test plan

1. **Unit test**: in Python (or torch), compute `w(x,y)` naïvely vs by recurrence over a 16×16 tile, for 100 random Gaussian conics. Assert max abs diff < 1e-5 (matches existing kernel atol).
2. **Numerical drift**: confirm error doesn't accumulate dangerously across 64-pixel scanlines (worst case for our tile sizes).
3. **CUDA microbench**: implement recurrence variant of forward kernel; benchmark on 3080 Ti and 4070 mobile at N=4096, H=540, W=960, F=8 (R=8 mode). Compare wall-clock vs current naïve `__expf` per pixel.
4. **End-to-end**: run on pico-002 training + inference; verify PSNR/LPIPS unchanged within noise band, ms reduction within claim.

## Acceptance gate

- Unit test: max |diff| < 1e-5
- Microbench: ≥1.5× speedup on rasterizer forward
- E2E: PSNR delta < 0.01 dB vs naïve at same checkpoint, training step time -10% to -25%

## Risks

- **Cumulative float drift** over very long scanlines (>256 pixels). Tile-bounded rendering caps this naturally; verify worst-case.
- **Branching cost** if we mix recurrence for medium Gaussians with LUT for narrow + naïve for large. Three-kernel dispatch overhead might eat the savings; benchmark.
- **Register pressure** from holding `w`, `r`, and the constant `exp(−a)`. Compare register allocator output for both kernels.

## Lab notes

### 2026-05-08 — math identity VALIDATED at fp64

**Test:** `tests/perf/test_h001_conic_row_recurrence.py` (8 unit tests)

- 100 random conic+center configurations × 5 seeds: max abs error vs naïve `np.exp` per-pixel evaluation < `1e-10` over 16×16 tile
- `Δ²q_x = 2a` constant-second-difference identity exact within `1e-12` for all sampled configs
- Isotropic-Gaussian column symmetry preserved exactly
- Wider 32×32 tile stress test: no negative-drift, no underflow→0 issues for `a ∈ [0.05, 0.5]`

**Conclusion:** the math is bit-exact at fp64. Hypothesis transitions `untested` → `partial`. CUDA port still pending — fp32/bf16 accumulation and SFU `__expf` rounding could change the absolute error band, but math identity itself is sound.

**Next step:** the existing forward kernel (`oss/cuda/src/rasterizer_fwd.cu` lines 203-227) currently does per-pixel `expf` inside the WMMA weight setup. The recurrence variant requires:

1. Compute `q0`, `w0 = __expf(-0.5f * q0)`, `r0 = __expf(-0.5f * delta_q0)`, `s = __expf(-a)` once per Gaussian-row
2. March the row with multiplies: `w *= r; r *= s`
3. Add CUDA test matching recurrence kernel against existing within fp32 tol on real Gaussian batches
4. nvbench microbench for ≥1.5× speedup

**Risks unchanged:** cumulative float drift on long scanlines (we tile-bound naturally → low risk), register pressure increase, dispatch overhead if conditional with LUT kernel.

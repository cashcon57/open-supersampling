# H002 — Low-Rank Latent Splat with Decoder-Only F-Channel

**Status:** `untested`
**Class:** Approximate factorization (quality-impact possible)
**Filed:** 2026-05-08
**Source:** Council convergent (GPT-5.5, Opus 4.7, Gemini 3.1)

## Claim

Replace per-Gaussian feature dimension `f_g ∈ ℝ^{64}` with a low-rank latent `z_g ∈ ℝ^R` (R = 4 or 8), with the projection back to F=64 (or directly to RGB) **pulled outside the rasterizer**:

```
f_g ≈ B · z_g,  B ∈ ℝ^{64×R}, z_g ∈ ℝ^R, R ≪ 64

Y(p) = Σ_g w_g(p) · f_g
     ≈ Σ_g w_g(p) · (B z_g)
     = B · ( Σ_g w_g(p) · z_g )
```

Or in normalized/residual form:
```
Z(p) = (Σ_g w_g(p) · z_g) / (ε + Σ_g w_g(p))
m(p) = Σ_g w_g(p)
ΔI(p) = φ_θ(Z(p), m(p), I_base(p), D(p), u(p))
```

**Rasterizer accumulates only R channels.** Decoder φ_θ is the only place where the full feature space (or RGB) lives.

## Performance claim

- **Payload reduction**: 64/R-fold reduction in raster accumulation work (8× at R=8, 16× at R=4)
- **Register pressure** drops accordingly → higher occupancy on 3080 Ti / 4070
- **Bandwidth**: ~50-87% reduction on per-Gaussian feature loads

Council estimate: 136ms → 10-15ms on the rasterizer alone (Gemini's projection at R=4 + register-spill elimination).

## Quality claim — **APPROXIMATION, NOT IDENTITY**

This is a **rank-R approximation**. Equivalence to original requires `B · pinv(B) ≈ I` on the support of `f_g`, i.e., **the network must learn an intrinsically R-rank feature subspace**.

For pico (F=64), intrinsic rank is plausibly 4-8 based on natural-image statistics. For larger feature spaces this assumption may fail.

## Test plan

1. **Rank analysis on existing v6.1-pico-001 checkpoint**: SVD of feat tensor across a batch; report top-R singular value energy. If `top-8 energy / total > 0.95` → R=8 is feasible.
2. **Bottleneck retraining ablation**: train pico-002 with `nn.Linear(64, R)` followed by `nn.Linear(R, 64)` between rasterizer accumulator and decoder. Compare PSNR/LPIPS at R=4, 8, 16, 32 vs R=64 baseline. Quality cliff identifies the right R.
3. **Direct-RGB variant** (Gemini's most aggressive): rasterizer emits RGB+confidence directly (R=4 with first 3 channels = RGB, 4th = confidence). Compare against latent-decoded R=4 variant.
4. **Microbench**: rasterizer forward at R=4, R=8, R=16 vs R=64 baseline.

## Acceptance gate

- R=8 latent → R=64 decode: PSNR within 0.05 dB of R=64 baseline, LPIPS within 0.02
- R=4 latent → R=64 decode: PSNR within 0.15 dB
- R=4 direct RGB+conf: within 0.3 dB (acceptable for Performance tier)
- Rasterizer microbench: ≥4× speedup at R=8, ≥8× at R=4

## Compose with

- **H001 conic recurrence** — multiplicative gain (8× × 2× = 16× combined estimate)
- **H003 raster-fusion** — same low-rank Z(p) feeds into the fusion path
- **H004 L2-resident state** — smaller per-Gaussian payload makes L2-fit easier

## Risks

- **Sub-rank-R features cliff**: if F=64 is genuinely high-rank, R=8 will lose visible quality. Pre-checkpoint SVD analysis is the cheap-precheck.
- **Decoder cost**: post-raster decoder `B z → 64 → RGB` adds work. Net win depends on decoder being cheap (1×1 conv class).
- **Direct-RGB variant** loses the latent flexibility (no per-pixel feature for downstream attention/fusion).

## Lab notes

(empty — untested as of 2026-05-08)

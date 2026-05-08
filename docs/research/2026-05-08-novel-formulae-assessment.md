# Novel Formulae Assessment — Phase 4 ms/Frame Optimization Council

**Date:** 2026-05-08
**Source:** Opus 4.7 novelty review of formulae generated during the Phase 4 model-council thread
**Purpose:** Catalog which formulae used in v6.2 architecture v4 are textbook-known vs likely-unpublished, so we know what's worth claiming as novel contribution.

---

## Standard / known (no novelty claim)

These are textbook or standard graphics/ML formulae used as building blocks:

1. **Gaussian warp covariance update** — `Σ' = J Σ Jᵀ + Δt·D`. Standard Kalman / continuous-time diffusion.
2. **Conic quadratic form for EWA** — `q_g(x,y) = a·dx² + 2b·dx·dy + d·dy²`. Standard EWA splatting (Zwicker 2002).
3. **Conic ↔ covariance relationship** — `Λ = Σ⁻¹ = [a b; b d]`, `Σ = (1/(ad−b²))·[d −b; −b a]`. Standard 2×2 SPD linear algebra.
4. **Energy-based pruning** — `E_g = α_g · (s_u·s_v) · ‖feat_g‖₂`. Variants in 3DGS compression literature.
5. **Kalman refinement** — `x̂_{t|t} = x̂_{t|t−1} + K_t(z_t − H·x̂_{t|t−1})`. Standard Kalman filter.

---

## Likely novel — formulae we should track as hypotheses

Five formulations Opus assessed as not-in-current-literature in this exact form. Each gets its own hypothesis file in `docs/research/hypotheses/` for tracking validation status.

| # | Hypothesis | File | Novelty class |
|---|-----------|------|---------------|
| 2.1 | Conic row-recurrence for EWA Gaussian weights | `H001-conic-row-recurrence.md` | **Most novel** — algorithmic identity not seen in GS/EWA papers |
| 2.2 | Low-rank latent splat with projection pulled outside the rasterizer (R-channel canvas + decoder-only F-channel) | `H002-low-rank-splat-contract.md` | Novel system-level formulation |
| 2.3 | Raster-fusion replacing pixel↔Gaussian cross-attention (canvas-as-K/V via EWA, no softmax) | `H003-raster-fusion-no-attn.md` | Novel formulation of fusion operator |
| 2.4 | L2-resident canvas + budgeted tile scheduler enforcing target ms/frame | `H004-l2-resident-budgeted-scheduler.md` | Novel combination for SR/extrapolation |
| 2.5 | Canvas-warp-only frame generation (no separate FG model) | `H005-canvas-warp-frame-gen.md` | Structurally different from DLSS/FSR FG |

---

## Engineering-novel (not new math, but novel composition)

These are smart engineering glue, not new equations:

- **Jacobian-free warp branch** on `|∇·u| < ε` (threshold applied to known formula)
- **Dynamic degradation ladder** adjusting R / K_tile / active tiles to meet budget (scheduling policy)
- **Disocclusion-only spawn at pixel center + advect by MV** (novel combination, not novel math)

These don't require formal hypothesis records but are documented in the v6.2 arch v4 spec.

---

## Guidance

Per OSS research-log discipline:

1. **Every hypothesis below gets a memo BEFORE it drives a v6.2 design decision.**
2. **Every test result goes into the corresponding hypothesis file as "Lab Notes" entries** — including failures.
3. **Status field** in each hypothesis: `untested` → `in-progress` → `validated` / `refuted` / `inconclusive`.
4. **Before any external publication** (paper, blog post, demo): cross-check claim against latest literature again. Novelty assessments age fast.
5. **If ablation refutes a claim**, mark the hypothesis as `refuted` and update v6.2 arch v4 spec accordingly. Do not silently drop.

# H003 — Raster-Fusion Replacing Pixel↔Gaussian Cross-Attention

**Status:** `untested`
**Class:** Architectural substitution (quality-impact possible)
**Filed:** 2026-05-08
**Source:** Council convergent (GPT-5.5, Opus 4.7); Gemini 3.1 most aggressive variant

## Claim

Replace pixel↔Gaussian cross-attention `Attn(Q_pixel, K_gauss, V_gauss)` with the rasterizer ITSELF as the fusion operator:

```
G(p) = (Σ_g w_g(p) · z_g) / (ε + Σ_g w_g(p))
m(p) = Σ_g w_g(p)

F'(p) = F(p) + ψ_θ([F(p), G(p), m(p), D(p), u(p)])
```

where `ψ_θ` is a small 1×1 conv (or 1×1 → depthwise 3×3 → 1×1).

**No discrete softmax over tokens.** The Gaussian EWA splat IS the weighting operator. Each pixel's "attention" over the canvas is computed by physically rasterizing the canvas at that pixel location — distance-weighted by the Gaussian conics.

For tiles flagged by the disocclusion mask (~5% of windows in typical gameplay), keep a local top-K=16 attention as a quality-mode upgrade.

## Performance claim

- **Cross-attention cost** at K=64 Gaussians × 2000 windows: ~200-500μs of pure overhead (per Opus)
- **Raster-fusion cost**: same as the Gaussian raster pass — already on the critical path, ψ_θ is ~1×1 conv class
- **Net**: cross-attn block deleted entirely from default hot path → 200-500μs saved

## Quality claim

Cross-attention provides discrete token-routing (each pixel can pick distinct K). Raster-fusion provides continuous geometric routing (each pixel sees Gaussians weighted by spatial proximity).

For SR + frame extrapolation, the spatial-proximity bias is **arguably more useful** than the discrete-token-routing flexibility — that's the architectural bet.

**Quality risk**: if cross-attention was doing important non-spatial routing (e.g., picking semantically-related Gaussians from elsewhere on the canvas), raster-fusion can't replicate that.

## Test plan

1. **Drop-in ablation on pico**: replace cross-attn block with raster-fusion + 1×1 conv decoder. Train pico-002a with raster-fusion, pico-002b with cross-attn baseline. Compare PSNR/LPIPS at matched compute.
2. **Disocclusion-tile attention add-back**: add local top-K=16 attention only on disocclusion-flagged tiles (< 5% windows). Measure quality delta vs raster-fusion-only.
3. **Microbench**: total ms for fusion stage in three configs (cross-attn baseline / raster-fusion only / raster-fusion + disocclusion top-K).

## Acceptance gate

- pico-002a (raster-fusion only) PSNR within 0.2 dB of cross-attn baseline at same step count
- pico-002a + disocclusion top-K within 0.05 dB of cross-attn baseline
- Fusion stage ms reduction ≥ 200μs

## Compose with

- **H002 low-rank splat** — same `Z(p)` feeds raster-fusion; if H002 ships, H003 is the natural fusion architecture
- **H004 budgeted scheduler** — raster-fusion is per-tile; the scheduler decides which tiles get the disocclusion-attention upgrade

## Risks

- Quality cliff if non-spatial cross-attn routing was load-bearing for pico-001's quality
- Decoder ψ_θ may need to be larger than 1×1 if it has to compensate for lost cross-attn flexibility — eats the savings
- Disocclusion-tile attention add-back is critical for Quality tier; without it, mode-A-only

## Lab notes

(empty — untested as of 2026-05-08)

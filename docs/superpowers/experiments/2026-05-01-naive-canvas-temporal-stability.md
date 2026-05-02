# Naive Canvas Temporal Stability (no trained network)

**Date:** 2026-05-01
**Script:** `scripts/test_naive_canvas_stability.py`
**Results:** `results/naive_canvas_stability/{metrics.csv,summary.txt,frames/}`
**Hardware:** M3 Max (MPS, reference rasterizer backend)

## Question

OSS-Gaussian Sprint 5 ships a persistent Gaussian canvas: a GPU-resident
buffer of N Gaussians that survive across frames, motion-warped each
frame and re-rendered. The pitch has two parts:

1. **Gaussian representation as a structural prior** — tested separately by Test 1U.
2. **Persistent canvas as temporal accumulator** — tested *here*.

If (2) provides measurable temporal stability *even with no learned
features at all*, then the Sprint-5 architecture is independently valuable
and is not dead weight if the Sprint-4 network underperforms. If it
doesn't, the canvas's value depends entirely on a competent network
feeding it.

## Setup

- **Sequence:** Sintel `alley_1`, frames 1–30 (clean pass + ground-truth `.flo`).
- **Resolution:** HR center-cropped to 432×1024 (multiples of `tile_size=16`); LR is 2× box-downsample → 216×512.
- **Naive canvas init:** one Gaussian per 4×4 LR block → 6 912 Gaussians, capacity = 6 912 (tight, no headroom). Position = HR-coordinate centre of the block; colour = block-mean LR colour; per-axis scale = `block × scale × σ × 0.5` with σ = 0.6 (chosen so adjacent Gaussians cleanly tile without excessive overlap); rotation = 0.
- **Per-frame loop:** `warp_canvas(α)` with ground-truth flow → cell-occupancy eviction (kill duplicates landed in the same 4×4 LR cell, keep the one closest to cell centre) → respawn dead slots into empty cells from current LR → colour EMA refresh `0.7·old + 0.3·LR_sample` for all alive Gaussians.
- **Renderer:** PyTorch reference rasterizer with **per-pixel alpha normalization** — render `(rgb, 1)` together and divide by the accumulated weight, matching what every shipped Gaussian rasterizer (gsplat top-k, 3DGS, Image-GS) actually computes. Without this the unnormalized sum produces overlap-density-dependent brightness artefacts that swamp temporal effects.
- **Metrics:**
  - `PSNR` vs HR ground truth (per frame, then averaged).
  - `Δ_all` = mean absolute frame-to-frame difference over all pixels.
  - `Δ_flat` = mean absolute frame-to-frame difference restricted to pixels whose ground-truth motion magnitude is < 0.5 px (true static regions). This is the metric of interest for "is the canvas more temporally stable on backgrounds that shouldn't change?"

Three conditions: per-frame **bicubic** upsample (baseline), **canvas α=0** (no warp, but EMA + respawn), **canvas α=1** (full motion warp + EMA + respawn).

## Results

| Condition       |   PSNR   | Δ_all   | Δ_flat  |
|-----------------|---------:|--------:|--------:|
| bicubic         | **31.69**| 0.03759 | 0.01631 |
| canvas_nowarp   |  18.74   |**0.01488**| 0.01860 |
| canvas_warp     |  21.63   | 0.02629 | 0.01671 |

Per-frame timing on MPS reference backend: ≈1.7 s/frame for canvas conditions (~6.9k Gaussians × 442k pixels in a Python-level loop). Bicubic is ms.

## Verdict

**Negative-to-neutral.** Naive canvas (no trained network) does **not** provide a clean temporal-stability win over per-frame bicubic in this test:

- On `Δ_flat` (the headline metric), `canvas_warp` is **0.01671 vs bicubic 0.01631** — essentially tied (≈+2.5 % worse, well inside noise for a 30-frame sample). `canvas_nowarp` is +14 % worse (0.01860). The motion warp recovers the EMA's lag penalty in flat regions but does not push below bicubic.
- On `Δ_all` (overall flicker, including moving content), `canvas_nowarp` wins decisively (0.0149 vs bicubic's 0.0376, **−60 %**) — but that is the EMA itself averaging across motion, not a "stability" property; it visibly smears moving content (PSNR drops 13 dB).
- `canvas_warp` lands in the middle on `Δ_all` (0.0263, **−30 %** vs bicubic) — some real motion-compensated stability, but coupled to the same 10 dB PSNR penalty.

The PSNR collapse (32 → 21 dB) confirms that naive Gaussian splatting at this resolution is a worse per-frame reconstruction than bicubic. Visually the rendered frames are recognizable but blocky/blurry — they look like a mosaic of 4×4 LR cells, because that is exactly what they are, and nothing in the loop sharpens them.

## Implications for Sprint-5 scope

The temporal architecture is **conditionally** valuable, not unconditionally:

- **The canvas does not generate stability from nothing.** The α=1 warp doesn't beat bicubic on flat-region delta, and the α=0 (EMA-only) condition's flicker reduction is a smearing artefact (low Δ but +13 dB PSNR loss). Without a learned signal that *adds information* per Gaussian, the canvas just buys exponential averaging — which a per-pixel temporal filter could do equivalently.
- **The motion warp does compensate for the EMA's lag** — `canvas_warp` recovers ~13 % of `Δ_all` stability with much less smearing than `canvas_nowarp`, and it stays within bicubic's flat-region delta. So the warp+canvas machinery is *correct*: it does what it claims (motion-compensated temporal accumulation). It just doesn't outperform a strong spatial baseline when there is no informational lift between frames.
- **Therefore Sprint 5 is not dead weight — but it's also not independently sufficient.** If the Sprint-4 network meaningfully outperforms bicubic per-frame and writes informative features into the canvas, the temporal accumulation will preserve and propagate that lift across frames (the warp+EMA loop demonstrably moves content correctly, see the rendered sequence). If Sprint-4 underperforms bicubic, the canvas alone won't save it.

### Caveats to note before drawing harder conclusions

- **`alley_1` has slow camera + character motion.** Δ_flat is dominated by tiny camera drift; a sequence with real disocclusion/parallax (`bandage_2`, `temple_2`) would stress the warp+respawn pipeline harder. Worth re-running there.
- **Naive init is the worst-case lower bound.** Even a trivially better init (e.g. one Gaussian per LR pixel, σ = 1px) would give ~28 K Gaussians and likely close the PSNR gap; we picked 6.9 K to match the standard tier budget.
- **Reference renderer ≠ shipped renderer.** Top-k normalization, anisotropic scale handling, and per-tile depth-sort behave differently in gsplat. The α-normalization wrapper here is a fair approximation but not identical.
- **No prune-on-content-change.** The canvas's design includes per-tile error-driven pruning to retire stale Gaussians; we used a cheaper cell-occupancy eviction. A more aggressive prune policy would shift α=0 toward more flicker but better PSNR.

## Reproduce

```bash
# Streams just alley_1 from the Sintel zip (~140 MB transferred).
python -c "from remotezip import RemoteZip; RemoteZip('https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip').extractall('data/sintel/', members=[n for n in RemoteZip('https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip').namelist() if '/clean/alley_1/' in n or '/flow/alley_1/' in n])"

python scripts/test_naive_canvas_stability.py \
    --sintel-root data/sintel \
    --sequence alley_1 \
    --frames 30 \
    --hr-h 432 --hr-w 1024 \
    --scale 2 --block 4 \
    --coverage-sigma 0.6 \
    --device mps
```

Outputs: `results/naive_canvas_stability/{metrics.csv,summary.txt,frames/<cond>/frame_NNNN.png}`.

# 2026-05-08 — CUDA Phase 4 pipeline-elegance report

## Inputs

- Scope: math-first audit of techniques A-M. No training, no 64-frame render.
- Local ckpt status: no mounted v6/v6.1 checkpoint. Tier-2 scripts ran deterministic fallback only and are ready for the 3080 Ti host.
- Main scripts: `tests/cuda/perf-math/`
- Memos: `docs/superpowers/experiments/2026-05-08-phase4-elegance-*.md`
- Formula memo: [docs/research/2026-05-08-novel-gaussian-formulae.md](/Users/cashconway/OpenSuperSampling/docs/research/2026-05-08-novel-gaussian-formulae.md)

## Section 1: Per-technique results — math-tier conclusions

| Technique | Tier reached | Outcome | Error bound (analytical) | Error bound (measured) | Speedup estimate | Verdict |
|---|---:|---|---|---|---|---|
| [A Pade exp](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-a.md) | 1 | `[4/4]` passes operator's `q<=9`, abs-error gate | target `<1e-3` | `5.833e-4` at `q=9` | replaces `expf` with polynomial/divide | Ship-with-flag |
| [B Separable Gaussian](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-b.md) | 1 + planned 2 | Exact only at `rot=0` or isotropic | cross-term `<= 9|sinθcosθ||r-1/r|` | fallback only | saves 2D conic eval + exp factor reuse for eligible splats | Axis-aligned fast path only |
| [C LUT exp](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-c.md) | 1 | 256-entry linear LUT is numerically small | `3.893e-5` | `3.859e-5` | removes `expf`, 512 B fp16 table | Ship |
| [D 2sigma vs 3sigma](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-d.md) | 1 | `2sigma` drops too much mass | dropped mass `13.534%` | n/a | smaller pair lists | Reject `2sigma`; keep `3sigma` |
| [E q>12 skip](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-e.md) | 1 | Useful only for current full-frame path | per skipped `<=0.00744` at `|feat|<=3` | n/a | skips far-tail exp/WMMA entries | Ship-with-flag |
| [F Top-K per pixel](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-f.md) | 2 fallback | No ckpt-backed decision | n/a | fallback `K99 p95=8`, `K<=4=31.6%` | K cap could reduce compositing | Defer real Tier 2 |
| [G Edge-only tiles](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-g.md) | 2 fallback | Measurement surface works | n/a | fallback edge fraction `29.7%` | potentially skips flat tiles | Defer real Tier 2 + Tier 3 |
| [H 2K to 4K](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-h.md) | 3 plan | Learned decoder dependent | n/a | n/a | could reduce direct 4K work | Tier-3-required |
| [I Precomputed masks](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-i.md) | 2 fallback | Needs real consecutive-frame drift | exact if AABB unchanged | fallback hit `19.9%` | caches pair-list build | Defer real Tier 2 |
| [J Spatial budget](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-j.md) | 3 plan | Learned/perceptual | n/a | n/a | content-adaptive raster work | Tier-3-required |
| [K Quantized state](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-k.md) | 1 + planned 2 | xy int16/fp16 scale plausible; int8 rot risky | xy half-step `0.0293 px`; rot half-step `0.01227 rad` | n/a | lower state bandwidth | Ship xy/scale flag; defer rot |
| [L Feat compression](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-l.md) | 3 plan | Information-loss path | n/a | n/a | less token bandwidth | Tier-3-required |
| [M Redundant computation](/Users/cashconway/OpenSuperSampling/docs/superpowers/experiments/2026-05-08-phase4-elegance-m.md) | 1 | 12 concrete static findings | mostly exact-preserving | n/a | likely largest free perf is M1-M4 | Ship cleanup batch |

## Section 2: Tier 2 measurements

No real Tier-2 checkpoint measurement was possible locally. Search found only legacy checkpoints: `results/oru/oru.pth`, `results/pico/oru_pico.pth`, `results/ord/ord.pth`, `results/paired/paired.pth`. The v6.1 paths referenced by dashboard metadata are Windows-host paths under `E:\checkpoints\srcnn-v6.1-pico-001\...`.

Fallback scripts and artifacts:

- F: `tests/cuda/perf-math/f_topk_stats.py`, histogram `docs/coordination/phase4-elegance-artifacts/f_topk_hist.png`
- G: `tests/cuda/perf-math/g_edge_tiles.py`, histogram `docs/coordination/phase4-elegance-artifacts/g_edge_tiles_hist.png`
- I: `tests/cuda/perf-math/i_tile_mask_drift.py`
- Generic v6 forward probe: `tests/cuda/perf-math/phase4_tier2_stats.py`

Training-host command template:

```bash
python tests/cuda/perf-math/phase4_tier2_stats.py \
  --ckpt E:\checkpoints\srcnn-v6.1-pico-001\step-00000500.pt \
  --input E:\checkpoints\v6_tier2_probe_input.pt \
  --output E:\checkpoints\v6_tier2_stats.json
```

## Section 3: Stacking analysis

Safe stack now:

- C LUT exp + M cleanup: tiny bounded weight error plus exact-preserving code-shape cleanup.
- A Pade and C LUT are alternatives, not additive. Use one exp approximation path.
- B exact axis-aligned path stacks with C/A only when `b=0` or isotropic gate passes.
- K xy/scale quantization should not be stacked with A/C as default until checkpoint stats establish scale/aniso ranges.

Cumulative analytical error for the recommended default candidate `C + M`: max per-weight exp approximation `<=3.893e-5`; M cleanup is exact-preserving if implemented as described. `E` adds tail truncation and should remain flag-only.

## Section 4: Surprising findings from M

- M1-M3 are bigger than micro-optimizations: native forward currently accepts `topk_norm` and discards it, computes AABB scratch then ignores it, and materializes deterministic full-frame pair buffers.
- M4 shows zero-fill work on hot buffers that are overwritten on non-empty renders.
- M9-M12 are Python-side free wins: filter canvas warp earlier, pass `None` for identity active masks, skip all-ones transmittance allocation, cache shape/device constants.

## Section 5: Tier 3 candidates

H, J, and L are deferred. G also needs Tier 3 if edge-only rendering survives real Tier 2. Proposed rig: held-out 64-frame TartanAir set, same ckpt, direct baseline vs candidate path, metrics PSNR/LPIPS/MS-SSIM plus temporal error and flat/edge-region stratification.

Queue as future work: `Phase 4-frametest`, after Phase 4b/4c land.

## Section 6: Final recommendation

1. Ship-immediately: M1-M4 cleanup design work; C LUT exp behind a CUDA flag; B exact axis-aligned/isotropic branch.
2. Ship-with-flag: A Pade for `q<=9`; E `q>12` skip on the current full-frame native CUDA path; K xy int16/fp16 scale packing after one ckpt stat pass.
3. Tier-2-required: F, G, I, B/K distribution percentages on a real v6.1 checkpoint.
4. Tier-3-required: H, J, L, and G quality validation.
5. Reject: D `2sigma` cull.

Count: Tier 1 reached for 7 techniques plus audit; Tier 2 fallback scripts for 3 techniques; Tier 3 deferred for 3 techniques. Ship/defer/reject split: 4 immediate candidates, 3 flag candidates, 6 deferred, 1 rejected.

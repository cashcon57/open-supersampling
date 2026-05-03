# OSS-SR — CNN Track (post-pivot)

**Date forked:** 2026-05-02
**Pivot trigger:** `docs/superpowers/experiments/2026-05-02-splats-cannot-SR-definitive.md`
**Branch:** `v0.2-dev` (continues; no separate fork branch)

## Why this track exists

Sprint 4 verified across five independent paths that the 2D Gaussian splat representation cannot do single-image super-resolution competitively against bicubic at our resource budget. The V0.5 architecture (lite-tier param net + 12K-param pixel-residual CNN) **does** beat bicubic by +0.84 to +2.08 dB across train + held-out scenes. The `zero+residual` ablation showed the splat path is explicitly ignored by the trained residual head — the SR work lives entirely in the CNN.

We rip the splat path out and ship the CNN as OSS-SR.

## Architecture (v0)

```
LR  (3, h, w)
depth (1, h, w)        ┐
motion (2, h, w)       │
normals (3, h, w)      ├──► encoder (small UNet) ──► (B, 64, h, w)
canvas_hint (3, h, w)  │
─────────────────────  │
                       │
LR upsampled bicubic ──┘
                                                                ▼
                                   ┌────────── CNN super-resolver ──────────┐
                                   │ 3-conv 3×3 padding 1, hidden=64        │
                                   │ Tail: PixelShuffle 2× → 3 channels     │
                                   └─────────────────────────────────────────┘
                                                                │
                                                                ▼
                                                       (3, 2h, 2w)  +  bicubic_HR
                                                                │
                                                            clamp(0,1)
```

The current V0.5 implementation is essentially this with extra steps. Cleanup steps:

1. **Drop the `GaussianParamNetwork` + `OutputHead` + `Rasterizer` from the SR forward path.** Keep the modules in `oss/gaussian/network/` for the OSS-RR track, but `oss/sr/` becomes the canonical home for the SR forward function.
2. **Replace `(rendered, lr_up) → residual` with `(lr, depth, motion, normals, lr_up) → HR`.** The residual-vs-direct framing was an artefact of the splat fight; OSS-SR predicts HR directly.
3. **Tier scaling on the CNN, not the splat net.** Pico (Steam Deck) → small CNN; Standard → bigger CNN with deeper convs. Same trainer wires it.
4. **Drop the bank, the K-per-tile, the gbuffer-bias module from the SR config space.** They don't exist in OSS-SR.

## What carries over from Sprint 4 untouched

- `oss/gaussian/data/lr_synthesis.py` — engine-aliased LR pipeline.
- `oss/gaussian/data/{sintel,tartanair,hypersim,srgd,mixed}.py` — dataset adapters.
- `oss/gaussian/train/train.py` — trainer loop, bicubic comparison, checkpointing, `--smoke-test`, wall-clock kill, multi-scene + held-out eval. Just wire the new model in.
- `scripts/held_out_scene_probe.py` — load a checkpoint and score on a held-out scene.
- All experiment memos and lab notebook discipline at `docs/papers/`.

## What gets retired

- Sprint 5 (persistent canvas) **in the SR context.** Sprint 5 only makes sense if the splat path encodes useful image content per frame, which it does not for SR. If OSS-Gaussian-RR succeeds, Sprint 5 may revive there.
- Sprint 6 (frame extrapolation) **in the SR context.** Same reasoning.
- The "single trained model fans out via prune/retrain" plan was already dropped to "lite/standard from scratch + distil to pico" — that decision now applies to the CNN, not the splat net.

## V1 candidates (post-MVP)

- **Temporal accumulation** — wire warped previous-frame HR output into the
  `canvas_hint` channel. Architecture is already 90% there: the 12-ch input
  layout reserves channels 9–12 for canvas_hint and we feed zeros today. One
  data-adapter change (return consecutive frame pairs) + one training-loop
  change (warp prev output → feed as canvas_hint) unlocks DLSS-style temporal
  accumulation. Expected lift: +2–5 dB based on DLSS 2 Quality vs bicubic
  numbers, which use the same trick. Same `warp(state, motion, α)` primitive
  used by the temporal-Gaussian track — see `oss-gaussian-temporal-track.md`
  §"Design constraint — interpolation-readiness". Order this AFTER the
  per-frame multi-day baseline so we can isolate the temporal contribution.
- **ESRGAN-style residual backbone** for higher-quality SR with the same
  G-buffer inputs.
- **Multi-scale training** (2× and 4×).
- **Cyberpunk capture** via OSSContribute once we have a deployed v0 model to
  feed it.

## OSS-Gaussian-RR (parallel track)

Splats DO work for denoising — D1 memo showed Image-GS at n=1000 beats OIDN on PSNR. The Gaussian-track infrastructure (renderer, bank, output head, network) is preserved and pivoted to the RR (ray-reconstruction / denoising) problem where the representation is known to fit. See `docs/superpowers/oss-gaussian-rr-track.md` for the plan once it lands.

## Open questions

1. Does V0.5 with the splat path **physically removed** (not just zero-input-channel-weighted) train as well or better? Test by re-implementing the head as `Conv2d(in=12, out=3, ...)` with PixelShuffle 2× tail and re-training. If margin shrinks vs current V0.5, the splat input — even ignored — was somehow useful (regularisation? channel-noise as a generalisation aid?). If margin holds or grows, splats really were dead weight.
2. What's the ceiling at standard tier with a real CNN backbone (ESRGAN-RRDB or SwinIR small)? Cheap to test once OSS-SR forks.
3. Does cross-game generalisation hold when we train multi-scene vs single-scene? Production run on multi-scene SRGD plateaued at ~28 dB at step 90K — same as single-scene — so apparently yes, but the V0.5 variant being tested there was effectively the CNN-only path.

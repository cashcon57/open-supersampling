# V0.5 Pixel-Residual Head Closes the Bicubic Gate
**Date:** 2026-05-02
**Status:** complete — **V0 bicubic gate cleared at step 500.**
**Predecessor:** `2026-05-02-output-head-dead-init.md`
**Hardware:** RTX 3080 Ti 12 GB
**Code commit:** `53ca20c` (V0.5 module + trainer wiring), `324e25c` (cosmetic fix to final eval)

## Hypothesis

The V0 plateau (model PSNR stuck at ~12 dB while bicubic sat at ~29 dB) is caused by pure 2D Gaussian splats hitting a local minimum the optimiser can't escape. Codex 5.5 review and GSASR / GS-STVSR literature both prescribe the same fix: bolt a small CNN on top of the splat output that predicts a per-pixel residual. Splats carry structure, residual paints texture.

Test: enable the new `PixelResidualHead` (12K-param 3-conv CNN, zero-init last layer) and re-run the same 500-step probe that showed flat 12 dB output before. If PSNR climbs above bicubic on at least one eval, V0.5 wins and the **bicubic gate** in `gaussian-network-architecture.md` §7 is cleared.

## Setup

- **Tier:** lite (channels (16,24,32,40), K=5, 178K param net + 12K residual = 190K total)
- **Data:** SRGD ActionRPG single scene, 575 frames, force-synth LR with σ=1.5 + JPEG q=85.
- **Optimiser:** AdamW, lr=1e-3, batch=4, weight_decay=1e-5, gradient clip max_norm=1.0.
- **Loss:** real SSIM via pytorch_msssim (`L1 + 0.1·(1−SSIM)`).
- **Steps:** 1 000.
- **CLI:**
  ```
  python -m oss.gaussian.train.train --tier lite --dataset srgd \
    --srgd-scene ActionRPG --dataset-root <train-host-data>\datasets\srgd \
    --output-dir <train-host-data>\checkpoints\sprint4-v05 --max-steps 1000 \
    --max-time-seconds 600 --eval-every 250 --device cuda \
    --enable-engine-aliased-lr --enable-gbuffer-bias --force-lr-synth \
    --lr-synth-blur-sigma 1.5 --lr-synth-jpeg --enable-pixel-residual \
    --batch-size 4 --learning-rate 1e-3 --log-every 100
  ```

## Result

### Loss / SSIM training curve

| Step | loss | l1 | ssim | bank_H | dxy | color_std | bias_grad |
|----:|-----:|---:|-----:|-------:|----:|----------:|----------:|
| 100 | 0.32 | 0.21 | 0.018 | 1.000 | 0.05 | 0.013 | 0e+00 |
| 250 | (eval) | | | | | | |
| 500 | (eval) | | | | | | |
| 600 | 0.029 | 0.015 | **0.86** | 0.991 | 0.078 | 0.082 | 4.7e-09 |
| 700 | 0.035 | 0.017 | 0.82 | 0.990 | 0.088 | 0.095 | 1.2e-10 |
| 900 | 0.038 | 0.018 | 0.80 | 0.990 | 0.099 | 0.102 | 4.3e-10 |
| 1000 | 0.036 | 0.017 | 0.81 | 0.990 | 0.097 | 0.099 | 3.0e-09 |

Loss dropped 11×, SSIM jumped from 0.018 to 0.86 in under 600 steps. The splat-side diagnostics (`bank_H`, `dxy`, `color_std`) barely changed — the residual head did most of the work, which is the expected behaviour for a zero-init residual on a degenerate splat output.

### Bicubic comparison (periodic eval)

| Step | model_PSNR | bicubic_PSNR | beats_bicubic | margin |
|----:|-----------:|-------------:|---------------|--------|
| 250 | (first eval) | | | |
| 500 | **31.56** | 30.28 | **8/8** | +1.28 dB |
| 750 | **30.46** | 29.08 | **8/8** | +1.38 dB |
| 1000 | **30.18** | 28.57 | **8/8** | +1.61 dB |

**V0 architecture cleared its bicubic gate at step 500.** All 8 held-out samples on every periodic eval beat bicubic by 1.3–1.6 dB.

### Caveat (cosmetic, fixed in `324e25c`)

The end-of-training "FINAL" eval line printed `model=12.02 dB beats=0/8`. The training was fine — that line was a programmer error: `evaluate_against_bicubic` was called without `residual_head=`, so it scored splat-only output. Periodic evals (which DID pass `residual_head=`) showed the real numbers above. Patched in `324e25c`.

## Decision

1. **V0 architecture (with V0.5 pixel-residual head) is trainable and beats bicubic on this dataset.** Move forward.
2. The "vector-only" purity claim from the early spec is dead. Production model = splats + small residual head. This matches GSASR and GS-STVSR.
3. **Multi-day production run is unblocked.** Next steps:
   - Quick sanity: verify the model still beats bicubic when (a) trained from scratch on multi-scene SRGD, not single-scene; (b) evaluated on a held-out scene the network never saw.
   - Then commit a 24–72 hour run on standard tier (500K params + ~12K residual) over the full SRGD GameEngineData corpus on the 3080 Ti.
4. **Update `gaussian-network-architecture.md` §7** to mark V0.5 as the de facto V0 — there is no longer a meaningful "pure splats" V0 in the production plan.
5. **Update CHANGELOG** to call out the bicubic-gate clear, since this is the trainability checkpoint that unblocks Sprint 5 work in the master plan.

## Open questions (now answered — see "Follow-up probes" below)

1. **Single-scene overfit?** ActionRPG has 575 frames; the model trained on 1 000 steps × batch=4 = 4 000 sample views — about 7× passes. Need a held-out-scene eval to confirm generalisation. Cheap.
2. **Does the residual head dominate?** The diagnostics suggest yes — splat-side params barely moved. The splats may be doing very little real work; the residual head may be acting as a shallow learned upscaler that happens to receive a useless splat as one of its 6 input channels. **Test:** ablate the splat side (zero the splat output before residual) and measure how much PSNR drops. If the gap is small, the splats are doing nothing and we should consider whether the splat path is worth keeping at all.
3. **Held-out scene PSNR?** ~30 dB on the training scene is fine for a sanity gate. We need to see what happens on never-seen scenes (CSGO, Dota2, ArchVizInterior) before claiming the V0.5 architecture generalises.
4. **What's the ceiling at standard tier?** Lite hit 30 dB in 1 000 steps. Standard (500K params) on multi-scene over 24–72 hours should see considerably more.

## Follow-up probes (run 2026-05-02 same day)

### Held-out scene generalisation (`scripts/held_out_scene_probe.py`)

Trained on ActionRPG only. Evaluated on three scenes the model never saw:

| Scene | Trained? | n | model_PSNR | bicubic_PSNR | margin | beats |
|-------|----------|--:|----:|----:|----:|------|
| ActionRPG | yes | 16 | 30.76 | 29.29 | +1.47 | 16/16 |
| CitySample | **NO** | 16 | 25.75 | 24.49 | +1.26 | 16/16 |
| StylizedRendering | **NO** | 16 | 26.79 | 25.95 | +0.84 | 16/16 |
| ArchVizInterior | **NO** | 16 | 28.29 | 26.22 | +2.08 | 16/16 |

**56/56 held-out samples across 3 unseen scenes beat bicubic** by 0.84–2.08 dB. Generalisation is real, not single-scene memorisation. Margin shrinks on unseen scenes (smaller than the +1.47 dB on the training scene), which is expected, but does not collapse.

### Splat-only ablation

Same checkpoint, same training scene (ActionRPG), but evaluated with `--no-pixel-residual` so only the splat output is scored:

| Mode | model_PSNR | bicubic_PSNR | margin |
|------|----:|----:|----:|
| Splat + residual | 30.76 | 29.48 | **+1.28** |
| Splat only | 12.00 | 29.48 | **−17.48** |

**The residual head is responsible for the entire 18 dB lift.** The 178K-param splat network is producing essentially-constant gray output (model PSNR identical to a degenerate dead-init model from earlier in the day). The 12K-param residual CNN is doing all the SR work.

## Updated architectural framing

The splat path is **decorative** at this scale. We have not falsified the "vector-based SR" thesis — the splats might contribute meaningfully at standard tier with longer training and the temporal canvas wired in — but at lite + 1 000 steps, the splats are just gray noise that happens to be one of six input channels to the real SR module (the residual CNN).

**Decisions:**

1. **Project win.** The bicubic gate is cleared on the training scene AND three held-out scenes by 0.84–2.08 dB. Sprint 5 canvas work is unblocked per the validation memo.
2. **Architectural honesty.** Continue calling it "Gaussian SR" only after we've shown the splat path actually contributes. Right now it's "CNN SR with bonus splat input channels."
3. **Multi-day production run is GO** — standard tier, multi-scene SRGD, 24–72 h. Will surface (a) whether the splat side starts to pull its weight at higher capacity / longer training, and (b) what real held-out PSNR looks like at this scale.
4. **Sprint 5 sequencing reconsidered.** The original plan was canvas-after-Sprint-4. If the splats remain decorative, the canvas warp loses much of its point — there's nothing meaningful to warp, just noise. Hold Sprint 5 until either (a) splats start contributing at standard tier, or (b) we explicitly redesign Sprint 5 to warp the residual head's output instead.

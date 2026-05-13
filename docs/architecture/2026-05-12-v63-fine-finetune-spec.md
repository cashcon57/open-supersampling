# v6.3-fine — Composite-head Fine-tune Validation Run

**Filed:** 2026-05-12
**Status:** spec, queued behind v6.2-pico-002 reaching step 100K (~2 days out)
**Driver:** middle-path decision between full v6.3 retrain (5 days) and v7-direct (skip the 2D bridge). v6.3-fine is the cheapest possible empirical validation that the v6.3 ingredients (magnitude scaling + canvas-aware aux loss) actually shift training behavior, not just inference output.

## One-line claim

Take the finished v6.2-pico-002 checkpoint, freeze HAT-Tiny, train ONLY the composite_head + a learnable canvas-fusion-scale scalar with a canvas-aware auxiliary loss for ~10K steps. If the canvas probe's `extrap-vs-SR diff` grows from the baseline 0.000225 to >0.005 during fine-tune AND held-out PSNR doesn't regress more than 0.5 dB, we have empirical evidence the v6.3 design works in a real training loop. Then commit to v7 directly.

## Why this is the right shape

The pre-training tests (`tests/sr/v6/test_cholesky_covariance_param.py`, `test_parent_child_spawner.py`) validate each v6.3 ingredient *in isolation*. The canvas-scale sweep (`scripts/sr_v6_canvas_scale_sweep.py` output for step-70000) proved inference-time scaling alone doesn't produce real OSS-FX motion. What we DON'T yet know:

- Will the model **learn to use the canvas** if the aux loss pushes on it during real training?
- Does the canvas have enough information content to be load-bearing if the composite_head learns appropriate weights for the scaled input?

A 6–12 hour fine-tune answers both. Cheaper than the 5-day full v6.3 retrain; more informative than unit tests alone.

## Configuration

| Knob | v6.2-pico-002 baseline | v6.3-fine |
| --- | --- | --- |
| HAT-Tiny backbone | trainable | **frozen** (all params requires_grad=False) |
| Composite_head | trainable | trainable |
| `canvas_fusion_scale` | n/a (implicit 1.0) | **new learnable scalar**, initialized to 50.0 |
| Rasterizer | trainable | frozen (no parameter changes mid-fine-tune) |
| Spawner / canvas state machinery | trainable | frozen (reset per trajectory as in normal forward) |
| GAN | active | **disabled** (no adversarial gradient; pure regression + aux) |
| Charbonnier loss | active | active |
| LPIPS loss | active | active |
| MS-VGG / Sobel / Wavelet / Temporal-consistency | active | active |
| `lambda_canvas_aux` | n/a | **0.1** (Charbonnier on `composite_head([0, canvas_hr * scale])` vs GT) |
| Steps | 100K (full run) | **10K** |
| LR | cosine warm-restart schedule | constant 1e-4 (fine-tune, not from scratch) |

## What the aux loss does

The forward pass produces:

```text
features_concat = cat([refined_hr (HAT output), canvas_hr * scale])  # (B, feat_dim + R, H, W)
delta_main = composite_head(features_concat)
out_main = bicubic_hr + delta_main

# Aux: same composite_head, zero refined_hr, keep canvas_hr * scale
features_zero_refined = cat([zeros_like(refined_hr), canvas_hr * scale])
delta_aux = composite_head(features_zero_refined)
out_aux = bicubic_hr + delta_aux

loss = standard_losses(out_main, gt) + lambda_canvas_aux * Charbonnier(out_aux, gt)
```

Both forward calls share the same `canvas_hr` tensor and the same composite_head weights, so the aux loss directly punishes the model when the canvas channels can't produce a sensible image alone. The model has two routes to satisfy aux: (a) increase canvas_fusion_scale so the canvas channels dominate the head's response, (b) shape the canvas_hr signal itself to carry real image content. Both are exactly what v6.3 was supposed to teach during training.

## Success criteria

After ~10K fine-tune steps:

1. **Held-out PSNR doesn't regress > 0.5 dB** vs pico-002 step-100K baseline.
2. **Canvas probe re-run on the fine-tuned ckpt:**
   - `zero canvas_hr at composite_head` should differ from `normal canvas_hr` by >0.1 dB PSNR (proof model is using the canvas).
   - `extrap-vs-SR pixel diff` should grow from baseline 0.000225 → **>0.005** (37×+ growth means motion-scaling at α<1 actually shifts output).
3. **canvas_fusion_scale parameter** should converge to >10 after 10K steps (if it stays at 50 unchanged, the optimizer didn't push it; if it drops below 1, the canvas was actively counterproductive).
4. **GAN-free training is stable** (no NaN, no loss explosion).

If criteria 1+2 pass: commit to v7 directly. The 2D ingredients work in a training loop; N-D extension is the only remaining unknown.

If criteria 1+2 fail: v6.3 design needs revision before v7. Most likely cause: canvas-hr signal is too noisy because the disocclusion spawner is too conservative. Spec the parent-child spawner before either v6.3 or v7 retrain.

## Compute

- ~10K steps × 4.5 s/step on 3080 Ti = **~12.5 hours** wall-clock with the trainer alone.
- With the held-out supervisor running between checkpoints (current arrangement), expect ~14 hours.
- Storage: 1 checkpoint every 1K steps × 68 MB = ~680 MB total. Trivial.

On cloud: ~5 H100-hours, $7 spot. Effectively free.

## Why not just spec this AS v6.3?

We're calling it v6.3-FINE specifically to distinguish it from the full v6.3 retrain (which would do this same architectural work plus train from scratch for 100K steps). The "fine" tag signals:

- It is NOT a teacher checkpoint suitable for distillation.
- It is NOT a comparable held-out baseline (the frozen-backbone constraint biases the held-out comparison).
- It IS a 1-day go/no-go on the v6.3 design before we commit 4 weeks of v7 engineering.

If criteria pass and we go directly to v7, this checkpoint is discarded after the test. If criteria fail, we re-spec.

## Dashboard

The fine-tune is internal-validation only; it does not get a public dashboard run name. The probe output JSON goes into `docs/research/hypotheses/H010-v63-fine-validation.md` as the durable record of the result.

## What follows if v6.3-fine passes

Phase 1 of the v7 spec (`docs/architecture/2026-05-12-v7-nd-gaussians-spec.md`): Python ref N-D rasterizer + tests. Land while pico-002 finishes / v6.3-fine runs. Once v6.3-fine confirms the 2D ingredients work, Phase 2-3 (v7 model wiring + first OSS-FX training run) begin.

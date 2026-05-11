# v6.3 Pico-003 — Canvas Utilization Spec

**Filed:** 2026-05-11
**Status:** spec, queued behind v6.2-pico-002 (currently at step ~55K, finishes at 100K)
**Driver:** [H009 — v6.2 canvas not utilized](../research/hypotheses/H009-canvas-not-utilized.md). Empirical proof that the composite_head puts effectively zero weight on canvas_hr at every measurable training stage of pico-002.

## One-line claim

v6.3 pico-003 keeps the v6.2 architecture but rebalances the fusion boundary so the canvas branch is *forced* to be load-bearing during training. Three changes, **all of them inference-free**, designed to flip the model from "canvas runs but isn't used" (current state) to "canvas carries measurable SR signal".

## Why this matters beyond the metric

- The Gaussian canvas's *unique* architectural value (vs a pixel-grid SR pipeline) is **(a)** OSS-FX frame extrapolation at α<1 with zero added compute, and **(b)** anti-aliasing-by-construction via covariance resampling. Both depend on the canvas actually carrying image content. A checkpoint where the canvas is decorative cannot ship OSS-FX.
- The cost-estimate memo (`docs/coordination/2026-05-11-heavy-cloud-training-cost-estimate.md`) frames the multi-cycle path to frontier quality at $75K–$450K. **Spending that compute on a model whose canvas pipeline is dead weight is not defensible.** v6.3 is the cheapest experiment that proves the canvas can be used at all.

## Hypothesis

If the magnitude mismatch (canvas_hr ~1e-3 abs-mean vs refined_hr ~1e-1) is the dominant reason the composite_head ignored the canvas during pico-002 training, then **rebalancing the input magnitudes at the fusion boundary** will be sufficient to force the canvas to be used. If a simple scalar pre-multiply does not work, the **add-fusion architecture change** is the next-cheapest intervention. The **canvas-aware auxiliary loss** runs alongside both as belt-and-suspenders.

## Three changes (priority order)

### Change A — fusion-boundary magnitude scaling

The cheapest intervention. Replace the current fusion line in `oss/sr/v6/model.py`:

```python
# v6.2 (current):
delta = self.composite_head(torch.cat([refined_hr, canvas_hr], dim=1))

# v6.3 (Change A):
canvas_hr_scaled = self.canvas_fusion_scale * canvas_hr  # learnable scalar or vector
delta = self.composite_head(torch.cat([refined_hr, canvas_hr_scaled], dim=1))
```

`canvas_fusion_scale` initializes to **50.0** (chosen to bring canvas_hr abs-mean from ~2e-3 to ~0.1, parity with refined_hr). Either a single scalar parameter (1 param) or a per-channel vector (16 params). Recommend **single scalar** for the first run — minimum surface area to test the hypothesis.

**Inference cost:** zero. The scalar folds into `composite_head[0].weight` at export. Free at runtime.

**Foldable export:** YES — at inference time, multiply the first conv's weight slice on the canvas channels by the trained scalar, set the scalar to 1.0, identity behavior.

**Risk:** if the model was correct to ignore the canvas (e.g. spawner state too noisy), this forces noise into the output and degrades quality. Mitigation: the scale is *learnable*; the model can decay it back if needed. But it now starts useful instead of dead.

### Change B — add-fusion architecture rewrite

The next-cheapest intervention. Replaces concat-fusion with additive residual:

```python
# v6.2 (current): concat then 3x3 conv
# in:  refined_hr  (B, feat_dim, H, W)         e.g. feat_dim=60
#      canvas_hr   (B, latent_rank, H, W)       e.g. latent_rank=16
# fused = composite_head[0](cat([refined_hr, canvas_hr], dim=1))
#         3x3 conv: 76 in -> 60 out -> ~41 KMACs/px

# v6.3 Change B: project canvas into refined's space, add, then 3x3
# canvas_proj = self.canvas_proj_conv(canvas_hr)   # 1x1 conv: 16 -> 60 -> ~1 KMACs/px
# fused = self.composite_head[0](refined_hr + canvas_proj)
#         3x3 conv: 60 in -> 60 out -> ~32 KMACs/px
```

**Inference cost:** `1×1 conv (16→60)` at HR adds ~1 KMACs/px; `3×3 conv (60→60)` instead of `3×3 conv (76→60)` saves ~8 KMACs/px. **Net cheaper.**

Initialization: `canvas_proj_conv.weight` initialized so its output's abs-mean roughly matches `refined_hr`'s. With canvas_hr abs-mean ~2e-3 and projection 16→60, initialize weights at `~3.0 / sqrt(16)` to land projection at ~0.05 abs-mean (half of refined_hr). Or just use a `LayerNorm` on canvas_hr before projection so its scale is automatic.

Skip Change B for the FIRST training run — only adopt if Change A fails to produce canvas utilization.

### Change C — canvas-aware auxiliary loss

A training-time-only loss that forces the canvas branch to carry SR signal independently:

```python
# At training time, in addition to the main composite_head output:
canvas_only_delta = self.composite_head(
    torch.cat([torch.zeros_like(refined_hr), canvas_hr_scaled], dim=1)
)
canvas_only_rgb = (bicubic_hr + canvas_only_delta).clamp(0, 1)
loss_canvas_aux = self.lambda_canvas_aux * F.l1_loss(canvas_only_rgb, hr_target)
```

`lambda_canvas_aux` starts at **0.1** (low enough not to dominate the main loss; high enough to break the canvas-ignored equilibrium).

**Inference cost:** zero. The aux head is the SAME composite_head used at inference; we just *also* run it once more with refined_hr=0 during training. At inference no extra pass happens.

Adopt Change C in the first training run regardless of A/B outcome. Costs nothing at deployment, breaks the zero-weight equilibrium even if Change A alone is too subtle.

## Training-config diff vs pico-002

Same as pico-002 except:

- `--canvas-fusion-scale-init 50.0` (new flag)
- `--lambda-canvas-aux 0.1` (new flag)
- Output dir: `<train-host-data>/checkpoints/srcnn-v6.3-pico-003`
- Dashboard run name: `srcnn-v6.3-pico-003`
- Everything else (backbone hat-tiny, fusion_mode concat, spawner_mode disocclusion, latent_rank 16, GAN config, dataset, scale, optimizer, schedule) unchanged.

## Success criteria

After v6.3 pico-003 reaches step ~50K (post-GAN-warmup stable state, same checkpoint where we probed pico-002):

1. **Re-run `scripts/sr_v6_canvas_probe.py` on the v6.3 checkpoint.**
   - `zero` mode PSNR must differ from `normal` mode PSNR by **> 0.2 dB**. (Confirms canvas is used.)
   - `stale-canvas` mode PSNR must differ from `normal` mode PSNR by **> 0.05 dB**. (Confirms canvas update matters.)
2. **Held-out PSNR / LPIPS on TartanAir oldtown** must not regress vs pico-002 by more than 0.3 dB PSNR / 0.005 LPIPS at the same step. (Confirms forcing canvas use does not damage the model.)
3. **Canvas-only mode** (the existing diagnostic) PSNR must exceed bicubic by **> 1 dB**. (Confirms canvas carries SR signal independently.)

If 1+3 pass and 2 holds, the canvas is now load-bearing AND we have not hurt SR quality. v6.3 is the new teacher.

If 1+3 pass but 2 regresses by more than the gate, the canvas-use intervention damaged quality — likely the spawner / canvas state is genuinely noisy on this footage. Spec a v6.3.1 with the spawn-aggressiveness knob (option #3 from H009 followup, which DOES cost frametime — accept it for the teacher only, the student distillation strips it back out).

## Inference cost summary

The three changes combined: **zero added inference frametime.** Change A folds into a conv weight at export. Change B is *cheaper* than the current path (~33 vs ~41 KMACs/px at HR). Change C is train-time-only.

This preserves the v6.2 spec's commitment to keep the teacher's architecture exportable to a sub-2-ms student via distillation.

## What this does NOT fix

- HAT-Tiny is still in the inference path of the *teacher*. The H006/H007 verdict (HAT-Tiny is too slow for end-user inference and must be replaced by a ≤1M student) is unaffected.
- The PSNR / LPIPS apples-to-oranges issues vs DLSS/FSR (different test data, different LPIPS backbones) are not addressed.
- OSS-FX α<1 frame extrapolation is enabled in principle by a working canvas but not implemented in code.

## Implementation plan

1. Land the three changes in `oss/sr/v6/model.py` behind config flags so v6.2 pico-002 ckpts still load (set defaults so `canvas_fusion_scale=1.0` and `lambda_canvas_aux=0.0` preserve v6.2 behavior).
2. Add the two new CLI flags to `scripts/sr_train_v6.py`.
3. Pico-002 finishes around step 100K (current 55K, ~3 days remaining on the 3080 Ti at 4.5 s/step).
4. Launch pico-003 from scratch (not warm-start from pico-002 — the magnitude rescaling needs to be learned in early steps, which a warm-start would skip).
5. Probe pico-003 at step 50K with the same `sr_v6_canvas_probe.py` harness, verify the success criteria above.

## Estimated compute

Identical to pico-002: ~5 days on RTX 3080 Ti to reach 100K steps at the current 4.5 s/step rate, ~125 GPU-hours total. ~$185 spot / ~$375 on-demand on a single H100 if we moved to cloud. Funds out of the same single-Heavy-run tier from the cost-estimate memo.

## What this DOESN'T require

- No retrain of v5 / v4.
- No new dataset.
- No worker / dashboard / R2 changes — pico-003 plugs into the existing RUN_CONFIG flow via a one-line entry in `scripts/build_public_dashboard.py`.
- No new GPU. Same 3080 Ti. Same supervisor. Same held-out eval pipeline.

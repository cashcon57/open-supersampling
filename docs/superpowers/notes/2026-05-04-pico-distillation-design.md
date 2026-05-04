# Pico-Tier Temporal Distillation Design

**Purpose:** Sprint 6 prep for a Steam Deck / integrated-GPU-friendly v5-pixel-temporal tier.

**Status:** Design memo only. Do not treat this as a trained model or ship decision.

## Target

The current v5-pixel-temporal standard model is about 626K params. That is too expensive for Steam Deck-class RDNA 2 and integrated GPUs without matrix acceleration. The pico target is a small student model trained to imitate the completed v5-pixel-temporal teacher.

Initial target:

- Parameter budget: about 150K or lower for the full temporal model.
- Runtime target: materially cheaper than standard tier, suitable for a Vulkan/HIP vector fallback path.
- Quality target: preserve most of the temporal teacher's perceptual/temporal benefit, accepting some PSNR loss if LPIPS and stability remain acceptable.

## Architecture Sketch

Proposed module: `oss/sr/temporal_pico.py`.

Interface should match `TemporalSRModel`:

```python
forward(
    lr_inputs,
    prev_hr,
    depth_hr_curr,
    depth_hr_prev,
    motion_lr,
) -> out_hr
```

Backbone:

- Use the existing SR-CNN pico tier from `oss/sr/cnn.py`.
- Source of truth: `SR_TIER_CONFIGS["pico"] == (16, 2)`.
- That means 16 hidden channels and 2 residual blocks for the single-frame SR backbone.

Temporal path:

- Keep the `DisocclusionGate` unchanged. It is only 3 learned scalars (`alpha`, `beta`, `gamma`) and is not a meaningful parameter or latency burden.
- Replace the standard `TemporalHead` hidden width of 32 with a smaller head, e.g. 16 hidden channels.
- Keep the same inputs to the head: current SR, warped previous HR, disocclusion mask, and HR depth.
- Keep first-frame behavior identical: bilinear previous-HR initialization from LR RGB.

Sketch:

```text
lr_inputs ── srcnn_for_tier("pico") ── current_sr
prev_hr + motion_lr ── warp_prev_hr ── warped_prev
depth_curr/depth_prev/motion ── DisocclusionGate ── mask
[current_sr, warped_prev, mask, depth] ── TemporalHeadPico(hidden=16) ── out_hr
```

## Distillation Objective

Teacher:

- Frozen completed v5-pixel-temporal checkpoint.
- Run in deterministic eval mode on the same frame pairs/windows used for pixel held-out and training.

Student:

- Pico temporal model.
- No direct GT supervision during distillation; the teacher output is the target.

Loss:

```text
L = L1(pico_out, teacher_out)
  + 0.1 * L_perceptual(pico_out, teacher_out)
```

Notes:

- Use LPIPS-VGG if available; use the existing lightweight proxy only for smoke tests.
- Do not include GT HR in the distillation phase. The goal is preserving the teacher's learned perceptual/temporal behavior in a cheaper model, not re-solving SR from raw labels.
- Keep temporal-consistency monitoring as an eval metric. Add it to the loss only after checking that it does not over-smooth the pico output.

## Training Data

Use the same TartanAir mix as v5-pixel-temporal.

Recommended pipeline:

1. Run teacher once over deterministic frame pairs/windows.
2. Save teacher `out_t` / `out_t+1` tensors or compressed image targets next to the sampled frame references.
3. Train pico against those offline teacher renders.

Why precompute:

- Avoids running the 626K-param teacher inside every pico training step.
- Makes distillation runs reproducible and cheap.
- Enables repeated pico architecture sweeps over the same targets.

## Schedule

Initial run:

- 50K steps.
- Batch size: as high as memory allows after teacher targets are precomputed.
- Estimated runtime: about 6 hours on RTX 3080 Ti.
- Optimizer: AdamW, start with the same LR family as v5-pixel-temporal, then tune.

Suggested phases:

1. 0-5K: freeze pico backbone, train the pico temporal head/gate around teacher targets.
2. 5K-40K: unfreeze all pico weights, full distillation loss.
3. 40K-50K: low-LR polish on the deterministic held-out-like TartanAir subset.

## Evaluation

Compare:

- v4 baseline.
- v5-pixel-temporal teacher.
- pico temporal student.
- bicubic.

Metrics:

- PSNR vs GT HR.
- LPIPS vs GT HR.
- Temporal stability using the same `t_motion` convention as the v5 scripts.
- Latency on RTX 3080 Ti, then Steam Deck / RDNA 2 fallback hardware.

Ship rule:

- Pico is not a replacement for the v5 winner unless it passes a separate quality/latency gate.
- It can ship as a low-power tier if it preserves enough of the teacher's LPIPS/stability improvement and materially improves latency.

## Open Questions

- Does pico need a smaller disocclusion gate, or are 3 scalar gates universal enough across tiers?
- Could pico use a more aggressive temporal accumulation strategy because per-frame cost is lower?
- How much teacher-target storage is acceptable for repeated distillation sweeps?
- Is a 16-channel temporal head sufficient, or does temporal blending need 24 channels even if the backbone is pico?
- Should teacher targets be stored as FP16 tensors, PNG/EXR, or regenerated on demand for exactness?

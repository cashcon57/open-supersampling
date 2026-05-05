# 2026-05-04 — v5 pixel-temporal warm-start distribution-shift bug

**Status:** Bug found mid-flight at step 14K. Run killed. Fix shipped (`b2fa647`). Training restarted. Step-1 loss went from 6.74 → 0.57 (12×).

## Symptom

While reviewing the live in-flight viz at `http://<tailnet-ip>:8080/`, Cash flagged: "v5-temporal still much worse than every other upscaling method." Visible chromatic dispersion / RGB streaking on tree foliage and overhead wires in the sample frames; LR-bilinear and bicubic both cleaner.

Train log at step 14,160:

```
step=14160 phase=2 loss=0.2407 t_l1=0.0406 tp1_l1=0.0431
                              t_lpips=0.4813 tp1_lpips=0.4763 tc=0.0667
```

t_lpips ~0.48 vs published bicubic LPIPS ~0.45 vs v4-alone ~0.30. Worse than bicubic on the perceptual metric. Worse than v4-alone.

## Root cause

`TemporalSRModel.forward()` ([oss/sr/temporal/model.py:59](oss/sr/temporal/model.py#L59)) called the v4 backbone with the **full 12-channel TartanAir input**:

```python
current_sr = self.backbone(lr_inputs)
```

where `lr_inputs = cat([lr(3), depth(1), motion(2), normals(3), canvas(3)])`.

The v4 backbone was trained on SRGD ([oss/gaussian/data/srgd.py:149-152](oss/gaussian/data/srgd.py#L149)):

```python
depth_lr   = torch.zeros((1, lr_h, lr_w))
motion_lr  = torch.zeros((2, lr_h, lr_w))
normals_lr = torch.zeros((3, lr_h, lr_w))
normals_lr[2] = 1.0  # default "up"
```

So the v4 backbone's conv1 weights against channels 3–11 were trained on a **constant signal** (all zero except `normals[2]=1.0`). Feeding TartanAir's real depth (metres → [0,1]), real optical flow (px), and depth-derived normals into those channels at warm-start time was a hard input-distribution shift. The conv1 weights against those channels did the only thing they could do given untested input: produce garbage.

Step-1 evidence (run that was killed): `t_l1 = 3.3355`. v4-alone on its training distribution sits at `t_l1 ≈ 0.005–0.01`. Off by ~500×. Backbone output was scrambled from frame 1.

The temporal head's near-passthrough init (`std=1e-3`) faithfully passed the scrambled backbone output through to the final SR. Phase 1 (10K steps, backbone frozen) couldn't fix it — the bad output was the backbone's, not the head's. Phase 2 (backbone unfrozen at step 10K) started corrupting the v4 weights with bad gradients from the confused temporal head on top.

Net result: 14K steps spent climbing out of a self-inflicted distribution hole.

## Fix

Match the SRGD training distribution: pass only RGB (+ normals[2]=1.0) to the backbone. Real G-buffers still feed the warp, disocclusion gate, and temporal head via their dedicated arguments.

```python
lr_for_backbone = torch.zeros_like(lr_inputs)
lr_for_backbone[:, :3] = lr_inputs[:, :3]
if lr_for_backbone.shape[1] >= 7:
    lr_for_backbone[:, 6] = 1.0  # normals[2] = SRGD default-up
current_sr = self.backbone(lr_for_backbone)
```

Mirrored in [`oss/sr/temporal/stateless_export.py`](oss/sr/temporal/stateless_export.py) for ONNX/inference parity.

## Verification

Smoke test on remote (CPU, fresh model, no checkpoint):

```
diff=0.0174  PASS
```

`diff = mean |TemporalSRModel.forward output  −  backbone(RGB-zeroed) output|`. Confirms output ≈ backbone-on-training-distribution at init, with tiny temporal head residual matching the std=1e-3 init.

## Before / after

|  | Pre-fix step 1 | Post-fix step 1 |
|---|---|---|
| `loss` | 6.7364 | 0.5652 |
| `t_l1` | 3.3355 | 0.2285 |
| `tp1_l1` | 3.2430 | 0.2298 |

Step-1 t_l1 ratio: **14.6×**. Pre-fix needed ~4,000 steps to reach a loss the post-fix run hits at step 1.

Trajectory of the restarted run (first 60 steps):

```
step=1  loss=0.5652
step=20 loss=0.5169
step=40 loss=0.4938
step=60 loss=0.4485
```

Smooth descent, no spikes, no NaN — model is in a sane region of weight space from frame 1.

## Cost paid

- ~2.5 wall-clock hours of training time on the broken run (steps 0 → ~14,300)
- 10 hours of restart wall-clock ahead

## What this rules in / out

**Rules in:** any future v5 architecture that warm-starts from a backbone trained on a different input distribution must explicitly handle the distribution mismatch. The pattern is general: warm-start ≠ free if the input pipeline changed.

**Rules out:** the design plan to "let the backbone adapt" via phase-2 joint training. It does adapt, but slowly and with collateral damage to v4 weights, and the resulting model is bounded above by what the bad gradients allowed it to learn.

## Followups

1. **v6 / Sprint 6 followup:** retrain a backbone end-to-end on real-G-buffer data, then drop the channel-zero hack. The backbone-with-real-G-buffers should beat backbone-with-zero-G-buffers given enough training. Out of scope for v5 — v5 has to ship a temporal SR result, not a new backbone.
2. **Codex review:** ask Codex to audit other sites that load v4 weights and pass non-RGB input. The held-out viz pipeline ([scripts/sr_temporal_inflight_viz.py](scripts/sr_temporal_inflight_viz.py)) uses `TemporalSRModel.forward` so it's already fixed transitively. But manual `backbone(...)` calls elsewhere may have the same bug.
3. **Test:** add a regression test that asserts `TemporalSRModel(...).forward(real_g_buffers)` ≈ `backbone(rgb_only_zeros)` ± temporal head init residual. Currently smoke-tested but not in the test suite.

# v5 Pixel Temporal Super-Resolution — Design Spec

**Status:** Design (control track in v5 dual-track race)
**Date:** 2026-05-04
**Author:** Cash Conway + Claude (Opus 4.7)
**Parent track:** [oss-sr-cnn-track.md](../oss-sr-cnn-track.md)

## Goal

Add FSR 2-class temporal accumulation to the v4 single-frame SR-CNN. Closes the perceptual quality gap to FSR 2 / DLSS 2 by reusing motion vectors and prior-frame HR output. Serves as the **control track** in the v5 dual-track experiment — the proven recipe against which the experimental Gaussian-temporal track is measured.

## Non-goals

- Multi-frame attention (that's the Gaussian track; pixel uses 1-frame history only)
- Frame extrapolation (separate track, OSS-FX)
- Vendor-specific kernel optimization (post-v5)
- Game integration / DLL hook (post-v5)

## Success criteria

Held-out fixed-batch eval on Sintel + a TartanAir held-out trajectory (not seen in training):

- [ ] PSNR ≥ +1.5 dB over v4 baseline
- [ ] LPIPS ≤ 0.20 (vs v4 ~0.31)
- [ ] Temporal stability: warp-then-diff between frames t and t+1 ≤ 0.5× the v4 single-frame variance
- [ ] No regression on bicubic-beats: ≥ 95% of held-out frames beat bicubic on both PSNR AND LPIPS

## Architecture

### Inputs (per frame)

```
LR color (3ch)        — current frame, low-res
Depth (1ch)           — current frame, LR
Motion vec (2ch)      — t-1 → t flow, LR
Normals (3ch)         — current frame, LR (derived from depth gradient)
Canvas hint (3ch)     — accumulator placeholder, LR
Prev HR output (3ch)  — frame t-1 SR result, HR
Prev disocclusion (1ch) — alpha mask from t-1 → t (HR)
```

Total: 12ch LR + 4ch HR. The HR-resolution prev-frame inputs are the new temporal axis.

### Network

```
[ LR inputs ] ──── existing v4 SR-CNN backbone ──── current_sr (HR)
                                                          │
[ Prev HR output ] ──── warp by motion vec ──── warped_prev (HR)
                                                          │
[ Prev disocclusion + warped_prev + current_sr ] ──── temporal head ──── final HR output
                                                          │
                                                          └─ disocclusion logit (HR)
```

**Components:**

1. **v4 backbone (frozen for first N steps, then unfrozen)** — produces current-frame single-frame SR estimate as before. Reuses checkpoint `srcnn-prod-v4-lpips/step-00385000.pt` exactly.
2. **Motion vec upsample** — current motion vec is LR; upsample to HR via bilinear (or bicubic; benchmark both).
3. **Backward warp** — `F.grid_sample(prev_hr_output, motion_grid, mode='bilinear', align_corners=False, padding_mode='border')`. Backward (pull) warp using the motion vec from t-1 → t.
4. **Disocclusion mask** — derived from depth disparity AND motion-vec magnitude:
   - `depth_diff = |warped_depth_prev - depth_curr|`
   - `motion_mag = ||motion||`
   - `disoccl = sigmoid(α·depth_diff + β·motion_mag - γ)` (learnable α, β, γ)
   - Mask is HR resolution.
5. **Temporal head** — small conv stack (~50K params):
   - Input: `concat(current_sr, warped_prev, disocclusion, depth_hr)` = 8 channels HR
   - 3× Conv(8→32, 3×3) + ReLU
   - 1× Conv(32→32, 3×3) + ReLU  
   - 1× Conv(32→3, 3×3) — final blend
   - Output: final HR frame
   - Optional: predict disocclusion logit as secondary output for loss supervision.

**Training-time first-frame init:** for the first frame in a sequence, `prev_hr_output = bilinear-upscale(current_lr)` and `prev_disocclusion = ones`. This makes the first frame degenerate to a slightly-lower-quality single-frame eval, which is fine; subsequent frames benefit from real temporal info.

### Inference state

- Maintain `prev_hr_output` as a single HR FP16 buffer in inference state (~24 MB for 4K)
- Reset on scene cut (detected by motion-vec magnitude or depth disparity exceeding threshold)
- Optional history clamp (FSR 2 trick): clamp `warped_prev` to local neighborhood color statistics in `current_sr` to suppress ghosting on rapid disocclusion

## Loss

```
L_total = L_appearance + λ_temporal · L_temporal_consistency

L_appearance = w_l1 · L1(out_t, gt_hr_t)
             + w_ssim · (1 - SSIM(out_t, gt_hr_t))
             + w_lpips · LPIPS(out_t, gt_hr_t)
                                   
L_temporal_consistency = w_tc · L1( warp(out_t, motion_t→t+1) · valid_mask,
                                    out_{t+1} · valid_mask )
```

**Hyperparameters (start point):**
- `w_l1 = 1.0`, `w_ssim = 0.1`, `w_lpips = 0.1` (same as v4)
- `λ_temporal = 0.05` (small — too high causes blurring)
- `valid_mask` = inverse of disocclusion (only penalize where warping is valid)

The temporal consistency loss requires sampling **two consecutive frames** per training step and rendering both. ~1.5× compute per step vs v4.

## Training

### Data

**Primary:** TartanAir Easy split, 18 environments, ~600 GB extracted, sequential trajectories with real flow + depth.
- Sequence sampling: pick a trajectory, pick a start index `i`, sample frames `(i, i+1)` at the same scene+camera state.
- Augmentation: random crop, horizontal flip (with motion-vec sign flip), brightness/contrast jitter (matched across the pair).

**Secondary:** Sintel clean pass — small (~1041 frames) but real Blender-rendered ground truth. Use as held-out validation.

**Tertiary:** SRGD scenes (where we have sequential frames but zero flow) — use only for fine-tuning where we approximate flow via RAFT (deferred; not in v5 scope).

### LR synthesis

Same `EngineAliasedLRSynth` (Halton jitter + TAA blur + JPEG q=85 + blur σ=1.5). Apply identically to both frames in a pair (with consistent jitter offsets) so temporal consistency loss isn't fighting LR-synthesis noise.

### Schedule

1. **Phase 1 — warm-up (steps 0–10K):** freeze v4 backbone, train only the temporal head + disocclusion params. Loss: appearance only (no temporal consistency yet). Establishes that the head can learn the warp+blend pattern.
2. **Phase 2 — joint (steps 10K–60K):** unfreeze backbone with reduced LR (10% of head LR). Add temporal consistency loss. Full loss active.
3. **Phase 3 — fine-tune on Sintel (steps 60K–80K):** small LR (1% of base), Sintel-only. Real-data polish.

Total: ~80K steps, ~12–16 hours on RTX 3080 Ti.

## Files

- **New module**: `oss/sr/temporal/`
  - `__init__.py` — public exports
  - `temporal_head.py` — temporal blend module (conv stack)
  - `warp.py` — motion-vec upsample + backward warp helpers
  - `disocclusion.py` — disocclusion mask computation
  - `dataset.py` — sequential frame pair loader (wraps existing TartanAir/Sintel datasets)
- **New script**: `scripts/sr_train_temporal.py` — temporal training entry (or extend `oss/gaussian/train/train.py` with `--temporal` flag)
- **New eval**: `scripts/sr_temporal_held_out.py` — fixed-batch eval reporting PSNR + LPIPS + temporal stability
- **Updated**: `oss/sr/inference.py` — add stateful inference path (carries `prev_hr_output` across calls)
- **Tests**:
  - `tests/sr/test_temporal_warp.py` — backward warp roundtrip
  - `tests/sr/test_temporal_dataset.py` — pair loader returns consecutive frames
  - `tests/sr/test_temporal_head.py` — module forward + backward + checkpoint roundtrip
  - `tests/sr/test_temporal_loss.py` — temporal consistency loss with synthetic moving rectangle

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Motion vec quality varies across TartanAir envs | Pre-validate flow with RAFT correlation on a sample; flag bad envs |
| Temporal consistency loss causes blurring | λ small (0.05); ablate at 0.01, 0.05, 0.1 |
| Disocclusion mask is unstable | Supervise mask with depth-disparity ground truth (TartanAir has it) |
| First-frame init is wrong on every sequence boundary | Random sequence-start sampling so model learns to handle it |
| Inference state buffer (~24 MB at 4K) blows VRAM | FP16 + tile-based inference if needed |
| Training is 1.5× slower per step | Smaller batch or shorter run; expected |

## Validation gates

Before declaring v5-pixel complete:

1. PSNR + LPIPS criteria above
2. Visual A/B on a 10-frame held-out clip (no flicker, no ghosting on disocclusion)
3. ONNX export passes (existing path; verify temporal state buffer correctly exposed)
4. TRT FP16 export and benchmark — latency ≤ 1.5× v4 baseline at same resolution

## Open questions

- Backward warp vs forward splat? Backward is standard (FSR 2). Forward is more accurate for occlusion but requires explicit hole-filling. **Default: backward.**
- Detach prev-frame output during training? Common practice (avoids backprop-through-time blowup). **Default: detach.**
- History length? **Default: 1 frame for v5.** v6 multi-frame is the Gaussian track's job.

## Out-of-scope (deferred to v6 or later)

- Multi-frame history (transformer attention over N prev frames)
- Sub-pixel jitter accumulation explicit modeling (FSR 2 does this; we let the model learn it)
- Adaptive disocclusion threshold (start with learnable scalar; iterate if static parameter is insufficient)
- Custom CUDA kernel for the warp + blend (post-v5; that's the perf pass)

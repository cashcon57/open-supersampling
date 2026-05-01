# OSS-Gaussian Frame Extrapolation — Design Note

**Status:** Sprint 6 design (sibling of `2026-05-01-gaussian-sprint-6-plan.md`).
**Track:** Gaussian (sits alongside the pixel-based OSS-Fx track until graduation).
**Last touched:** 2026-05-01.

This document explains *why* OSS-Gaussian's frame extrapolation works the way
it does, how it differs from DLSS Frame Generation, and where it fails.

---

## 1. The trick in one sentence

The persistent Gaussian canvas already warps to follow motion every real
frame. Frame extrapolation reuses that warp with a fractional alpha to
produce the intermediate frame. There is no second model.

Concretely (Sprint 5 produces the canvas; Sprint 6 reuses it):

```python
# Sprint 5 — runs every real render at frame t
canvas.gaussians = warp_canvas(canvas, motion_t_minus_1_to_t, alpha=1.0).gaussians
canvas.spawn_on_disocclusion(...)
canvas.prune_high_error(...)

# Sprint 6 — runs between real renders at fractional alpha
intermediate = extrapolator.extrapolate(canvas, motion, alpha=0.5, output_hw)
```

The cost is one in-place add on the (N, 2) Gaussian-position tensor. The
rasteriser does the same work it would for a real frame; everything
upstream is bypassed.

## 2. How this differs from DLSS Frame Generation

| Property | DLSS Frame Generation | OSS-Gaussian Frame Extrapolation |
|---|---|---|
| Pipeline shape | Separate optical-flow net + per-frame fusion pass | Same warp the canvas does anyway, with a smaller alpha |
| Model size | ~10 MB optical-flow + fusion network on tensor cores | 0 — no model |
| Latency cost vs base render | Adds a heavy GPU pass (~3–4 ms on 4090) | One position add (~µs) |
| What can fail | Optical flow model is wrong → ghosting | Linear warp at alpha=1 wrong → ghosting |
| Variable cadence | Fixed 1:1 (one synthesised frame per real frame) | Any alpha schedule (60→90, 60→120, 60→144 supported) |
| Hardware tier scaling | Requires Ada-class tensor cores | Scales with Gaussian count knob; works on any GPU that runs the rasteriser |
| Reflex / latency | Adds frame-time variance because flow net is the bottleneck | Adds essentially zero latency above base render |

DLSS-FG is genuinely solving a harder problem (it must invent pixels with
no prior structure), and on RTX 4090 it produces beautiful results. The
OSS-Gaussian approach takes a structural shortcut: because the canvas
already encodes the scene as motion-warpable primitives, we just shift
them less. The shortcut works only because Sprint 5 did the hard work
upstream of producing a temporally-stable canvas.

## 3. Algorithm

```
Inputs:
    canvas:  PersistentCanvas at time t (Sprint 5 output)
    motion:  (2, H, W) per-pixel motion field for t-1 → t
    alpha:   scalar in [0, 1], 0 = current frame, 1 = predicted next frame
    output_hw: (H, W) target render resolution

Algorithm:
    1. warped = warp_canvas(canvas, motion, alpha=alpha)
    2. return rasterizer(warped.gaussians, output_hw)
```

Step 1 mutates only positions; covariance, rotation, and color are reused
unchanged (per design spec §3.2: "covariance frozen"). Step 2 is the
exact rasteriser path Sprint 1 ships and Sprint 5 already calls every
real frame.

## 4. Alpha schedules

For an integer FPS ratio ``target_fps / source_fps``, reduce to coprime
form via `gcd`. The scheduler emits the synthesised alphas uniformly
across the displayed period (real frames at alpha=0 are implicit).

| Source | Target | gcd | Real / period | Synth / period | Alphas |
|---|---|---|---|---|---|
| 60 | 90 | 30 | 2 | 1 | `[0.5]` |
| 60 | 120 | 60 | 1 | 1 | `[0.5]` |
| 60 | 144 | 12 | 5 | 7 | `[0.4, 0.8, 0.5, 0.83, 0.4, 0.83, 0.5]` (uniform-ish) |
| 60 | 240 | 60 | 1 | 3 | `[0.25, 0.5, 0.75]` |

The 60→120 case is canonical (frame doubling), and the test suite
verifies alpha=0.5 is the only synthesised alpha in that schedule.

## 5. Failure modes

### 5.1 Non-linear motion at high alpha

The warp model is *linear*: position(α) = position(0) + α × motion.
For projectile arcs, accelerating vehicles, or whip-camera-pan, this
under- or over-shoots. The error scales with α². At α=0.5 (60→120) the
error is ¼ of the α=1 error, which is why we recommend that cadence as
the default.

**Mitigation:** none in v1. Quality degrades gracefully — predicted
frames look slightly stale or slightly leading rather than catastrophically
wrong. Sprint 5 disocclusion handling catches the worst cases by
spawning fresh Gaussians on the next real frame.

### 5.2 Fast rotation

Angular velocity > ~120 °/s produces motion fields where adjacent
pixels' motion vectors diverge significantly across the image. The
average-motion approximation in our test double becomes wrong; the
real Sprint 5 warp does per-Gaussian lookups so it handles rotation
correctly *as long as the motion field is dense*. If the motion field
itself is sparse / blurred (common in Cyberpunk's lower-resolution MV
output), high alpha rotations look smeary.

**Mitigation:** clamp alpha to ≤ 0.5 when the motion field's per-pixel
variance exceeds a threshold (Sprint 6 stretch goal — not in v1).

### 5.3 Disocclusion at high alpha

A Gaussian spawned at frame t never existed at t-1. Its predicted
intermediate position is meaningless. Sprint 5 T5.5 tags spawn-this-frame
Gaussians; Sprint 6 T6.5 freezes them at α=0 position regardless of the
caller's alpha so they do not contribute ghost trails.

**Mitigation:** spawn-flag tensor honoured by `FrameExtrapolator`.

### 5.4 Camera teleport / scene cut

Motion field becomes meaningless across a hard cut. Both DLSS-FG and
OSS-Gaussian fail here; both rely on the integration code skipping
extrapolation when the engine flags a cut.

**Mitigation:** Sprint 5's interception layer detects scene cuts via
G-buffer correlation and signals the canvas to skip warp + extrapolation
for one frame.

## 6. Latency budget vs DLSS-FG

For an 8K-Gaussian canvas at 1440p on a 3080 Ti (Sprint 1 bench targets):

| Stage | DLSS-FG (1440p, Ada-equivalent) | OSS-Gaussian |
|---|---|---|
| Base render | n/a — runs in parallel | ~1.7 ms (Sprint 1 T1.6 estimate) |
| Optical flow | ~1.2 ms | 0 ms |
| Fusion pass | ~2.0 ms | 0 ms |
| Warp | n/a | <0.05 ms (one tensor add) |
| Rasterise intermediate | n/a | ~1.7 ms |
| **Total intermediate cost** | **~3.2 ms** | **~1.75 ms** |

The OSS-Gaussian intermediate effectively costs one extra rasteriser
pass — half a real render — versus DLSS-FG's ~3 ms additive overhead.
This is the budget headroom that lets us consider 60→144 cadences
without falling off the GPU's frame budget.

(Numbers above are projections; T6.2 and T6.4 produce real measurements
on the 3080 Ti.)

## 7. Expected quality

We do not yet have measurements. The numbers below are predictions made
to set acceptance thresholds for Sprint 6 T6.3 / T6.4:

| Alpha | Expected PSNR vs ground-truth | Notes |
|---|---|---|
| 0.0 | ∞ (identity) | Sanity check; always passes. |
| 0.25 | 38–42 dB | Low warp magnitude; non-linearity error tiny. |
| 0.5 | 32–36 dB | Canonical 60→120 case. Acceptance threshold ≥ 32 dB. |
| 0.75 | 28–32 dB | Visible degradation in fast scenes. |
| 1.0 | 25–30 dB | Equivalent to next-frame prediction; comparable to a low-quality optical-flow predictor. |

These predictions assume Sprint 5 produces a temporally-stable canvas
with PSNR-vs-OSSPico parity (its own graduation criterion). If Sprint 5
falls short, all numbers above shift down proportionally and the
graduation decision is informed accordingly.

## 8. Graduation contribution

Sprint 6 contributes to the spec §5 graduation criteria as follows:

- **PSNR + SSIM:** baseline measurements at α=0.5 → bar that DLSS-FG
  must beat to keep its place as the comparison reference.
- **Temporal stability:** ghosting metric (frame-to-frame pixel delta in
  flat regions of intermediates) measured during T6.3.
- **Subjective:** T6.4 produces side-by-side video of OSS-Gaussian vs
  DLSS-FG at 60→120; user makes the call.
- **Latency:** T6.2 + T6.4 produce hard frame-time numbers; if
  OSS-Gaussian is within 110% of DLSS-FG the parity criterion is met.

Sprint 6 does not gate graduation on its own — the full Cyberpunk
Sprint 5 + Sprint 6 pipeline does, in concert with Sprint 7's
cross-platform validation.

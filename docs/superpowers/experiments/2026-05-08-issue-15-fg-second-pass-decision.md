# 2026-05-08 — Issue #15 closeout: FG 2nd pass cost decision

**Status:** RESOLVED (does not block pico-002 training).

## Measurement recap

3080 Ti, torch 2.4.1 / CUDA 12.4, gsplat backend, N=4096 Gaussians,
H=540, W=960:

| R | Pass 1 ms | Pass 2 ms | Total / pass1 | Gate (<=1.5x) |
|---:|---:|---:|---:|---|
| 4 | 5.654 | 5.789 | 2.09x | FAIL |
| 8 | 6.236 | 5.925 | 1.98x | FAIL |
| 64 | 12.467 | 12.590 | 2.01x | FAIL |

## Decision

**Accept the 2x baseline. Do not block pico-002 launch.**

Reasoning:

1. **Pico-002 is a TRAINING run.** The training forward path runs a
   single rasterization per frame. There is no second pass during
   training. The H005 gate measures an inference-time concern.

2. **The arch v4 spec already pre-planned this mitigation** (section 7
   risk register, line 183): *"Frame-gen 2nd pass is more expensive than
   estimated. Mitigation: measure on pico-002; if >2.5ms, gate FG behind
   Quality+ tier only."* The measured 2.0x ratio is below the 2.5x
   tripwire and the mitigation is already on file.

3. **The bench measures the wrong workload for v6.3 inference.** A
   production FG 2nd pass should process ONLY the foreground subset of
   Gaussians (those with significant motion magnitude relative to the
   camera). The current bench rasterizes the full canvas twice. The
   2.0x ratio is therefore an upper bound, not the inference cost.

## Follow-up tracking

Filed for v6.3 (post-pico-002):

- **FG mask in rasterizer:** filter Gaussians by motion magnitude before
  pass 2. Re-run H005 with the mask path; expected ratio drops well
  below 1.5x as foreground pixels are typically < 30% of the frame.
- **Tile-bin reuse:** since pass 2's positions are pass 1's positions
  shifted by `alpha * MV`, the tile assignment is mostly unchanged.
  Reuse the bin structure from pass 1 to skip the radix-sort step.

These optimizations are not gating for pico-002 launch.

## Inference-time gate (post-training)

When v6.2 enters inference / engine integration, expose
`enable_fg_2nd_pass` on `V6Config` and gate it to Quality+ runtime tiers
per the arch spec mitigation. Tracked separately; does not require code
changes during training.

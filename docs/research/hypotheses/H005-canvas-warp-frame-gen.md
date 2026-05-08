# H005 — Canvas-Warp-Only Frame Generation (No Separate FG Model)

**Status:** `partially-validated` (architecture runs; not benchmarked vs DLSS-FG / FSR-FG)
**Class:** Architectural — structural property of the canvas design
**Filed:** 2026-05-08
**Source:** OSS architecture (since v5); explicitly characterized by GPT-5.5 + Opus 4.7 + Gemini 3.1 in 2026-05-08 council

## Claim

Frame extrapolation in OSS is a **deterministic side-effect of canvas advection**, not a separately-trained model.

For an extrapolated frame at fractional time `α ∈ (0, 1)`:

```
Base reprojection:
  I_{t+α}^base(p) = I_t(p − α · u_t(p))

Canvas extrapolation:
  μ_{g, t+α} = μ_{g, t} + α · u_t(μ_g)
  Λ_{g, t+α} = (Jacobian-free if ∇·u < ε, else A^-T Λ A^-1)

Hole / edge correction:
  I_{t+α}(p) = I_{t+α}^base(p) + M_hole(p) · ΔI_gauss(p)
```

**Cost:** second warp + raster pass (~1.4-2ms additional on reference hardware), with no separate optical-flow accelerator and no separate AI model.

Compare to DLSS-FG / FSR-FG, which require:
- Optical flow accelerator (DLSS-FG, dedicated silicon)
- Separately-trained FG network (consumes compute + memory + training pipeline)
- Multiple frames + flow input per generated frame

## Performance claim

OSS frame extrapolation cost ≈ rasterizer pass × 2 (≈ 1.4-2ms additional)
DLSS-FG cost on Ada ≈ 1-2ms (plus optical-flow accelerator hardware)
FSR-FG cost ≈ 2-4ms (software optical flow)

**OSS structural advantage**: no optical flow network → frame-gen scaling is linear in number of frames generated (canvas warp at α=0.5, 0.66, 0.75 etc. for MFG modes).

## Quality claim

Canvas advection is **deterministic** — same canvas + same motion field → same extrapolated frame. No AI-hallucinated frames. Quality bounded by:
- Quality of motion vectors (engine-provided)
- Coverage of canvas Gaussians (more Gaussians → fewer holes)
- Correctness of Jacobian-free vs full warp branch

## Status: partially-validated

What's validated:
- Architecture runs end-to-end through pico-001 training (frames are produced via canvas warp)
- Canvas warp + reproject + raster does produce extrapolated frames without an additional model

What's NOT validated:
- Quality vs DLSS-FG / FSR-FG in apples-to-apples conditions
- Cost vs DLSS-FG (need ms measurements on matching hardware)
- Multi-frame gen (MFG 2× / 3× / 4×) cost scaling claim
- Hole correction quality at high α (large extrapolation distance)

## Test plan

1. **MFG cost scaling**: implement frame-gen at α = 0.5 (FG 1×), {0.33, 0.66} (FG 2×), {0.25, 0.5, 0.75} (FG 3×) on pico-002. Measure ms cost per generated frame. Verify linear scaling.
2. **Quality at large α**: visual + LPIPS at α = 0.5 vs α = 0.75 vs α = 0.9. Identify the α-cliff where canvas advection diverges from ground truth.
3. **Apples-to-apples vs DLSS-FG**: integrate OSS into a Unity / Unreal demo, compare against DLSS-FG on RTX 4070 in same scene. Same source FPS, same target FPS, measured ms + PSNR + LPIPS.
4. **Hole correction**: when canvas Gaussians don't cover a region, the hole-correction MLP fires. Measure quality of holes vs DLSS-FG hole filling.
5. **Disocclusion artifact comparison**: DLSS-FG known to have ghosting on fast disocclusions. OSS canvas approach has different failure mode — measure both.

## Acceptance gate

- MFG 2× cost ≤ 2.5× FG 1× cost (linear-ish scaling)
- α = 0.5 quality within LPIPS 0.02 of ground-truth interpolated frame
- vs DLSS-FG on 4070: within 0.5 dB PSNR at matching ms budget

## Compose with

- All of v6.2 architecture (H005 IS the architectural property; H001-H004 are kernel/scheduling support)

## Risks

- Quality at α > 0.5 may degrade visibly (motion field linearization breaks down)
- Canvas coverage gaps cause holes that hole-correction MLP can't always fix cleanly
- Multi-frame gen with our canvas approach may stack errors faster than DLSS-MFG (which uses fresh AI inference per generated frame)

## Lab notes

- 2026-05-08 — Architecture runs; pico-001 trains; frames produced via canvas warp. No separate FG model exists in pipeline. ✅ Partial validation of structural property. NOT yet benchmarked.

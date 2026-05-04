# 2026-05-04 — v3 vs v4-LPIPS fixed-batch A/B

**Status:** Result
**Question:** Is v4 (L1+SSIM+0.1·LPIPS-VGG, forked from v3 step-240K) a real improvement over v3 (L1+SSIM only)?
**Why it matters:** v5 temporal builds on top of one of these checkpoints. If v4 isn't actually better, we'd be extending a phantom regression. Per-eval random-batch numbers during v4 training showed direction but had ±1-2 dB sample variance — needed a fixed-batch test to be sure.

## Method

Wrote `scripts/sr_v3_vs_v4_ab.py`:

- Loads two checkpoints (`ckpt-a` v3 step-240K, `ckpt-b` v4-lpips step-385K — best LPIPS during training)
- Runs both on the **same deterministic 64-sample batch** from CitySample (held-out scene during training), `torch.manual_seed(0)`, `shuffle=False`
- Reports PSNR + LPIPS for each model and bicubic baseline, plus per-sample win counts

Same `EngineAliasedLRSynth` (Halton jitter + TAA blur σ=1.5 + JPEG q=85) as training.

## Result

```
=== A/B fixed-batch eval (CitySample, n=64) ===
  ckpt_a = step-00240000.pt   (v3, L1+SSIM)
  ckpt_b = step-00385000.pt   (v4, L1+SSIM+0.1·LPIPS)

PSNR (dB, higher is better)
  A          : 29.565
  B          : 29.530
  bicubic    : 25.810
  B-vs-A     : -0.035 dB
  A>bicubic  : 64/64
  B>bicubic  : 64/64
  B>A        : 24/64

LPIPS-VGG (lower is better)
  A          : 0.4036
  B          : 0.3147
  bicubic    : 0.5103
  B-vs-A     : -0.0890  (-22.0%)
  A<bicubic  : 64/64
  B<bicubic  : 64/64
  B<A        : 64/64
```

## Reading the numbers

- **PSNR: tied.** −0.035 dB delta is well below per-sample variance (~0.5–1 dB on 8-frame batches we saw during training). 24/64 B>A on PSNR is essentially a coin flip — confirms statistical tie, not v4 regression.
- **LPIPS: v4 wins unanimously.** −22.0% relative, **64/64 frames perceptually preferred for v4**. This is the cleanest signal possible — every single frame, not just on-average.
- **vs bicubic:** both models beat bicubic 64/64 on both PSNR and LPIPS. Healthy floor.

## Conclusion

v4 is a real perceptual improvement over v3, paid for by a statistical-tie PSNR cost. This matches Real-ESRGAN's reported LPIPS-loss tradeoff almost exactly: trade a small PSNR for a large perceptual win.

**Decision:** v4 step-385K is the production single-frame baseline. v5-pixel-temporal will warm-start from this checkpoint. v5-gaussian-temporal will use it as a comparison reference (shares the V0.5 splat infrastructure starting point, not v4 weights directly).

## Caveats / honest limits

- **Single scene** (CitySample) on **one held-out batch**. Other scenes might show different ratios; we should re-run on Sintel + a TartanAir held-out trajectory once those are extracted.
- **LPIPS-VGG is one perceptual metric.** Has known biases. Real perceptual eval requires either DISTS, FID-on-game-content, or human raters — not done here.
- v3 ran with motion-vec/depth/normals all zeros (SRGD has no real G-buffers). v4 fork ran identically. This A/B doesn't tell us anything about how either model would behave with real motion vectors — that's the v5 question.

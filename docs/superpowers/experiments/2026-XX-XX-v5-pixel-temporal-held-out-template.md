# YYYY-MM-DD — v5-pixel-temporal held-out vs v4

**Status:** Result
**Question:** Does v5-pixel-temporal meet the success criteria from the design spec?

## Method

Same fixed-batch protocol as v3-vs-v4. Loaders shuffle=False, torch.manual_seed(0).
Sintel held-out clean pass + TartanAir held-out trajectory.

## Results

(paste output of scripts/sr_temporal_held_out.py here)

## Success criteria

- [ ] PSNR ≥ +1.5 dB over v4 baseline
- [ ] LPIPS ≤ 0.20 (vs v4 ~0.31)
- [ ] Temporal stability ≤ 0.5× v4 single-frame variance
- [ ] ≥ 95% held-out frames beat bicubic on PSNR AND LPIPS

## Conclusion

(write decision: ship v5 / iterate / fall back)

## Caveats / honest limits

- Single fixed-batch eval; rerun on additional scenes if any criterion is borderline.
- LPIPS-VGG is one perceptual metric.

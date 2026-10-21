# H011 — v7 Thin-Structure Error Heatmap Residuals

**Status:** `observation`
**Class:** Quality / diagnostics — v7 OSS-FX error-map interpretation
**Filed:** 2026-06-08
**Source:** Manual dashboard viz inspection of `srcnn-v7.0-pico-005` around step ~19K.

## Claim

Bright crossbar-like structures in the `|err v7 alpha=0.5|` dashboard panel can appear around window mullions even when the non-error `v7 alpha=0.5` RGB panel does **not** visibly contain duplicated mullions/crossbars.

Interpretation: this is most likely a thin-structure **edge-alignment / contrast residual** in the absolute-error heatmap, not a model hallucination of extra window geometry.

## Observation

Cash flagged a v7 viz crop where the `|err v7 alpha=0.5|` panel showed extra horizontal/vertical crossbeam-like lines across windows/brick facade. Re-checking the corresponding non-error `v7 alpha=0.5` RGB panel showed no visible duplicate crossbars.

The artifact is therefore in the diagnostic residual visualization:

```text
err_panel = abs(v7_alpha_0_5 - GT-half)
```

Thin, high-contrast mullions/window frames make absolute-error maps visually overstate small spatial/contrast differences. A 1-2 px edge offset, width mismatch, blur mismatch, or contrast mismatch can render as paired/doubled residual contours even when the RGB prediction looks acceptable.

## Status: observation, not blocker

This is **not** currently evidence that v7 is learning wrong geometry.

It is evidence that v7 has measurable residual around high-frequency thin structures in the alpha=0.5 frame target. That is expected to some degree and should be judged against RGB quality, temporal stability, and aggregate metrics rather than the error heatmap alone.

## What would make it a real issue

Escalate from observation to model-quality issue if any of these become true:

1. The non-error `v7 alpha=0.5` RGB panel visibly shows duplicated mullions/crossbars.
2. The artifact flickers/crawls in playback rather than only appearing as a static error-map residual.
3. The residual grows over training steps or correlates with worsening LPIPS/temporal metrics.
4. The same pattern appears across many thin-structure scenes, not just a single facade/window crop.

## Follow-up checks

1. Compare `v7 alpha=0.5` directly against `GT-half` at 1:1 for the same row/step.
2. Watch the same facade across future milestone strips (`20K`, `21K`, ...), focusing on RGB flicker/duplicate geometry, not just heatmap brightness.
3. If needed, add a thin-structure slice metric: edge-mask or line-structure residual over high-gradient GT regions, tracked separately from full-frame PSNR/LPIPS.

## Lab notes

- 2026-06-08 — Dashboard viz step ~19K: `|err v7 alpha=0.5|` showed crossbeam/mullion-like bright residuals over a window facade. Non-error `v7 alpha=0.5` did not show the same duplicated geometry. Current classification: **thin-structure edge residual in error map, not model hallucination; not a training blocker by itself.**

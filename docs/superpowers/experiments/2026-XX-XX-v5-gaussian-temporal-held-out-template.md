# YYYY-MM-DD — v5-gaussian-temporal held-out vs v5-pixel-temporal vs v4

**Status:** Result
**Question:** Does v5-gaussian-temporal meet the success criteria from the
design spec, AND does it explicitly beat v5-pixel-temporal on the same
fixed batch?

## Method

Same fixed-batch protocol as the pixel-temporal held-out (and v3-vs-v4 A/B).
Loaders `shuffle=False`, `torch.manual_seed(0)`. Sintel held-out clean pass +
TartanAir held-out trajectory. Three models scored on the SAME deterministic
batch:

- **G** — v5-gaussian-temporal (stateful Gaussian-field engine)
- **P** — v5-pixel-temporal (pixel-warp, prev_hr cold-start regime)
- **A** — v4 single-frame baseline

Flow-direction convention: when warping `out_t` to align with `t+1`, motion
fed in is `t_motion` (the forward flow `t -> t+1` lives at frame `t`), NOT
`tp1_motion` (which is `t+1 -> t+2`). Mirrors the pixel held-out fix at
commit `38cf507`.

The Gaussian engine is stateful (B=1). Each held-out pair is treated as an
independent two-frame stream: engine reset → frame `t` (seeds the field via
first-frame densification) → frame `t+1` (scored output).

## Results

(paste output of `scripts/sr_gaussian_temporal_held_out.py` here)

## Success criteria (Gaussian-temporal spec)

The four gates from the v5-gaussian-temporal design spec:

- [ ] **PSNR** ≥ pixel − 0.3 dB (i.e. Gaussian PSNR within 0.3 dB of pixel,
      and ≥ +1.5 dB over v4 baseline)
- [ ] **LPIPS** ≤ pixel − 0.01 (i.e. Gaussian perceptual quality is at least
      0.01 better than pixel, and ≤ 0.20 absolute)
- [ ] **Temporal stability** ≤ pixel (i.e. Gaussian flicker-metric is no
      worse than pixel; spec also requires ≤ 0.5× v4 single-frame variance)
- [ ] **Latency** ≤ 1.5× pixel (held-out script does not measure latency
      directly; verify via `scripts/bench_*.py` against the same checkpoint)

## Race rule (ship decision)

> **Gaussian must explicitly beat pixel; tie ≠ Gaussian win.**

Per-sample comparisons use **strict inequalities**:

- PSNR: `G > P` (strict greater-than)
- LPIPS: `G < P` (strict less-than)
- Temporal stability: `G < P` (strict less-than, ratio `G/P < 1.0`)

A tie on any criterion does NOT count as a Gaussian win. If Gaussian fails
to strictly beat pixel on ≥ 2 of the 3 quality axes, the ship decision is
"ship pixel; defer Gaussian to v6" — see Task 14 closeout for the formal
gate.

## Conclusion

(write decision: ship Gaussian / ship pixel / neither passes — iterate)

## Caveats / honest limits

- Single fixed-batch eval; rerun on additional scenes if any criterion is
  borderline.
- LPIPS-VGG is one perceptual metric.
- Gaussian engine cold-start: each pair starts from an empty field at
  frame `t`. Real deployment streams accumulate longer history; the
  held-out two-frame protocol may understate the temporal advantage of the
  Gaussian track in long sequences. If the held-out is borderline, run a
  longer-stream variant before falling back.
- Latency gate not exercised by this script — verify separately.

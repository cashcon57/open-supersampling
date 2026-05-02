# OutputHead Dead-Init Symmetry Failure
**Date:** 2026-05-02
**Status:** complete — root cause identified; fix queued
**Predecessor:** `2026-05-02-pico-lite-aggressive-srgd.md`
**Hardware:** RTX 3080 Ti 12 GB
**Code commit:** `5668590` (diagnostic metrics added)

## Hypothesis

The pico/lite training failures (flat 11–12 dB across 12K–20K steps) are caused by one of:
1. Bank softmax collapse to a single entry.
2. Position deltas saturating at zero (Gaussians stuck at tile centers).
3. Color sigmoid trapped near 0.5 (constant-gray output).
4. Architecture fundamentally incapable.

Diagnostic metrics added (`bank_entropy_norm`, `mean_dxy_norm`, `mean_color_*`, `color_std`) to disambiguate before committing to V0.5 pixel-residual work.

## Setup

- **Tier:** lite (channels (16,24,32,40), K=5, 178K params)
- **Data:** SRGD ActionRPG, single scene, 575 frames.
- **LR-synth:** σ=1.5, JPEG q=85 (smoke-test aggressive).
- **Optimiser:** AdamW lr=1e-4, batch=4.
- **Steps:** 500 (cheap probe).
- **Code commit:** `5668590`.
- **CLI:**
  ```
  python -m oss.gaussian.train.train --tier lite --dataset srgd --srgd-scene ActionRPG \
    --dataset-root <train-host-data>\datasets\srgd --output-dir <train-host-data>\checkpoints\sprint4-diag \
    --max-steps 500 --max-time-seconds 600 --eval-every 250 --device cuda \
    --enable-engine-aliased-lr --enable-gbuffer-bias --force-lr-synth \
    --lr-synth-blur-sigma 1.5 --lr-synth-jpeg --batch-size 4 --learning-rate 1e-4 \
    --log-every 50
  ```

## Result

| Step | loss | l1 | ssim | bank_H | dxy | color_std |
|----:|----:|---:|-----:|-------:|----:|----------:|
|  50 | 0.32 | 0.22 | 0.011 | **1.000** | **0.000** | **0.011** |
| 100 | 0.30 | 0.20 | 0.020 | 1.000 | 0.000 | 0.012 |
| 200 | 0.31 | 0.21 | 0.014 | 1.000 | 0.000 | 0.018 |
| 300 | 0.31 | 0.22 | 0.022 | 1.000 | 0.000 | 0.016 |
| 400 | 0.29 | 0.19 | 0.018 | 1.000 | 0.000 | 0.019 |
| 500 | 0.32 | 0.22 | 0.013 | 1.000 | 0.000 | 0.025 |

| Step | model_PSNR | bicubic_PSNR | beats_bicubic |
|----:|-----------:|-------------:|---------------|
| 250 | 11.52 | 29.22 | 0/8 |
| 500 | 11.50 | 28.99 | 0/8 |

**Every diagnostic is pinned at its degenerate value across 500 steps:**
- `bank_H=1.000` (max possible, perfectly uniform softmax across all 16 entries)
- `dxy=0.000` (Gaussians stuck at tile centers, no position delta)
- `color_std≈0.015` (colors near-identical across thousands of Gaussians → ~constant gray output)

This is **not** a capacity problem (lite has 178K params training on 575 frames — radically over-parameterised for this scale). It's a **dead-init symmetry failure**.

## Root cause

`OutputHead`'s output projection is **zero-initialised** (`oss/gaussian/network/output_head.py`). For K Gaussians per tile, all K parallel "decoder slots" start with identical raw outputs:

- Same `d_xy = (0, 0)` → same position (tile center).
- Same `bank_logits = 0` → same uniform softmax over the bank.
- Same `color = sigmoid(0) = 0.5` → same gray.

The renderer composites K identical Gaussians and produces a single gray blob per tile. The loss gradient is symmetric across the K branches; AdamW updates them in lockstep; **the symmetry never breaks.** This is the textbook failure mode for a model where K parallel decoders must specialise.

The CUDA gradient probe (`scripts/probe_cuda_grad_flow.py`) showed non-zero gradients on `head.weight` — it just didn't show that those gradients are perfectly symmetric across the K=5 branches. They are.

## Decision

**Fix the OutputHead initialisation:**

1. **Replace zero-init with small Gaussian random init** on `head.proj.weight` (and the gbuffer-bias projection if it stays). σ ≈ 0.01 is enough to break symmetry without destabilising early training.
2. **Add per-K-index positional encoding** to the head input so each of the K decoders sees a different feature even at init. Either: (a) concatenate a one-hot K-index channel before the head, or (b) use K parallel projection weights from a single learned bias matrix indexed by K.
3. **Re-run the 500-step probe.** If `bank_H` drops below 1.0 and `dxy` rises above 0 within 100 steps, the dead-init was the bug. If diagnostics still pinned, search deeper.

The G-buffer-bias module's zero-init is correct as-is (its job is to be a no-op at init and learn from gradients). The main head's zero-init is the bug.

**Defer V0.5 pixel-residual head until this is verified.** A pixel-residual head added on top of a dead-init splat output would mask the symmetry failure rather than fix it.

## Open questions

1. After fixing init, does the model start to learn (PSNR climbs) within 1K steps?
2. Is the K-index positional encoding necessary, or does small Gaussian init alone suffice? Test (a) Gaussian-init only, (b) Gaussian-init + positional, compare convergence speed.
3. Once the model trains, does the bank actually specialise (entropy drops, but to which entries)? That's the eventual ablation for `bank_size ∈ {8, 16, 32}`.

## Follow-up: smoking gun #2 (silent-zero CUDA backward) and the local-minimum plateau

After the K-symmetry-breaking bias init landed (commit `6900300`), a 300-step probe with `_param_health` logging revealed:

```
step=160 bias_grad=0.0000e+00 w_grad=0.0000e+00
step=180 bias_grad=0.0000e+00 w_grad=0.0000e+00
step=200 bias_grad=0.0000e+00 w_grad=0.0000e+00
step=220 bias_grad=0.0000e+00 w_grad=0.0000e+00
step=240 bias_grad=8.0009e-04 w_grad=9.2484e-03  ← non-zero ONCE
step=260 bias_grad=0.0000e+00 w_grad=0.0000e+00
step=280 bias_grad=0.0000e+00 w_grad=0.0000e+00
```

`net.head.bias.grad` was zero on ~13 of 14 logged steps. **Root cause #2: gsplat 1.4.0's CUDA backward returns silent-zero gradients when Gaussians are too small to hit any tile after coordinate normalisation.** Bank entry 0 (σ=1px isotropic) at scale_factor=exp(0)=1 produces ~1/256 normalised size on a 256-wide LR — consistently misses every tile. The CHANGELOG known-issue list flagged 2 failing CUDA-backward tests in this exact regime.

**Fix:** add `log(8) ≈ 2.08` to the bias for each Gaussian's `log_scale` channel (commit `6c02cc8`), so `scale_factor` starts at ~8 and Gaussians cover several tiles at init.

After both fixes (`6900300` + `6c02cc8`) plus bumping lr to 1e-2:

| Step | bank_H | dxy | color_std | bias_abs | model_PSNR |
|----:|-------:|----:|----------:|---------:|-----------:|
| 50 | 1.000 | 0.05 | 0.01 | 0.13 | 11.5 |
| 250 | 0.490 | 0.820 | 0.182 | 0.306 | 11.97 |
| 500 | 0.490 | 0.820 | 0.182 | 0.306 | 11.50 |

Diagnostics MOVED from init values for the first time. Network actually started picking bank entries (entropy halved), Gaussians moved well off tile centers (dxy=0.82), colors became varied (std 18× higher), parameters drifted 2.4×.

**But: model PSNR still 11–12 dB at every step. The model converged to a non-trivial-looking but still-degenerate local minimum that is no better than the constant-gray output.**

Loss landscape diagnosis: pure 2D Gaussian splats with K=5 per tile + fixed bank + L1+SSIM loss find a local minimum where Gaussians are "doing something" (not collapsed) but the splatting average produces low-frequency mush that is dominated by bicubic. The optimiser can't climb out to a regime that beats bicubic on PSNR.

**This validates Codex 5.5's V0.5 prescription.** Pure splats at this Gaussian budget + this loss cannot represent SR detail; a pixel-residual head on top of the splat raster is required to recover high-frequency texture. GSASR and GS-STVSR both use exactly this pattern.

## Updated decision

1. **Init fixes stay** (`6900300` + `6c02cc8`). They're correct — they just don't solve the local-minimum-plateau problem.
2. **Implement V0.5 pixel-residual head next.** Tiny CNN (2–3 conv layers) on the rendered HR output that predicts a residual; final output = `splat_render + residual`. Splats carry structure; CNN paints texture.
3. **Hyperparameter sweeps continue to be deprioritised.** lr ∈ {1e-4, 1e-3, 1e-2} all hit the same plateau. The fix is architectural.

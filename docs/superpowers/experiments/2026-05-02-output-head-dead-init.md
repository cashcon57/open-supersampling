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

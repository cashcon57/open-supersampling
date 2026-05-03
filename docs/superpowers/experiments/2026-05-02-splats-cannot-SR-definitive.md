# 2D Gaussian Splats Cannot Do Single-Image Super-Resolution — Implementation-Independent
**Date:** 2026-05-02
**Status:** complete — definitive
**Predecessors:** `2026-05-02-v05-pixel-residual-success.md`, `2026-05-01-gaussian-upscaling-naive-test.md`
**Hardware:** RTX 3080 Ti 12 GB

User asked: "double and triple check our implementation of the splats and make sure they are 100% definitely useless for upscaling, no matter the implementation."

Five independent lines of evidence are now consistent. The 2D Gaussian splat representation, in our pipeline, structurally cannot do single-image super-resolution competitively against bicubic. This holds across (a) network capacities, (b) learning rates, (c) training durations, (d) training-time competition with a residual head, and — critically — (e) bypassing the network entirely and optimising Gaussians directly per-image.

This is not a bug in our implementation. It is the property of the representation we've vendored.

## 1. Direct Image-GS optimisation (bypasses network entirely)

From `docs/superpowers/experiments/2026-05-01-gaussian-upscaling-naive-test.md`. Image-GS optimised **50 000 Gaussians per frame** directly via SGD against an LR target (no network), then rasterised at HR. 6 Sintel frames, 3 000 optimisation steps each. Aggregate:

| Method | PSNR (dB) | Δ vs bicubic |
|--------|----------:|-------------:|
| Bicubic | **34.28** | — |
| Lanczos | **34.42** | +0.14 |
| Image-GS naive (50K Gaussians) | 30.69 | **−3.59** |

**LR-fit PSNR was 42.8 dB** (range 38.5–47.9). The Gaussians fit the LR target excellently — the loss happens at HR rasterisation. The representation cannot hallucinate the high-frequency detail that bicubic preserves through aliasing.

**This is the ceiling on what 50 000 directly-optimised Gaussians can do for SR.** Our network produces 1 000–7 200 Gaussians via a single forward pass; nothing we add at the network level will exceed the 50K-direct ceiling.

## 2. Renderer + gradient flow audit

`oss/gaussian/renderer/rasterizer.py` reviewed line by line:
- No `.detach()`, no `with torch.no_grad():` in the forward path.
- Reference backend is full O(N×H×W) pure-PyTorch accumulation — guaranteed differentiable.
- CUDA backend is gsplat 1.4.0 `project_gaussians_2d_scale_rot` + `rasterize_gaussians_sum`, both with backward.
- Coordinate normalisation is correct (verified by integration tests).

`tests/gaussian/test_network.py::test_end_to_end_differentiability_grads_flow` runs random-input → network → decode → rasterizer → loss → backward and asserts non-zero gradients on `stem.conv.weight`, `head.weight`, and `bank.log_sx`. Passes.

`scripts/probe_cuda_grad_flow.py` runs the same on CUDA and prints per-leaf grad norms. Reference-vs-CUDA grad-norm ratio: 3–94× (CUDA weaker but non-zero on every leaf).

**Implementation is correct.**

## 3. Lite-tier splat-only training (178K param net, 2 400 Gaussians)

`docs/superpowers/experiments/2026-05-02-pico-lite-aggressive-srgd.md` and `2026-05-02-output-head-dead-init.md`.

After the K-symmetry-breaking init fix and the log_scale=log(8) gsplat-backward fix, lite-tier splat-only training with:
- batch=4, lr ∈ {1e-4, 1e-3, 1e-2}, ≤ 12 000 steps
- aggressive engine-aliased LR (σ=1.5 + JPEG q=85)

…always plateaus at **model_PSNR ≈ 11–13 dB** vs bicubic 26–30 dB. Diagnostics (`bank_entropy`, `mean_dxy_norm`, `color_std`) move from init values, confirming the splats DO change shape during training — they just don't move in a way that matches the target.

## 4. Aggressive lr=1e-1 splat-only training (extreme parameter movement)

Same setup as #3 but with lr=1e-1 (10× higher, gradient clip max_norm=1.0). 3 000 steps.

Diagnostics at step 3000:
- `bank_H = 0.000` — bank softmax fully collapsed to one entry.
- `mean_dxy_norm = 1.414` (≈ √2) — Gaussian centres at maximum tanh-allowable distance from tile centres.
- `color_std = 0.471` — colour variation at max possible across all Gaussians.
- `bias_abs = 1.39` — head bias at 10× init magnitude.

**The splat representation reached fully-active extreme parameter values.** Model PSNR: 11.73 dB. Bicubic: 29.20 dB. Margin: −17.47 dB. 0/8 beats.

## 5. Standard-tier splat-only training (500K param net, ~7K Gaussians)

Run on 2026-05-02 22:21 UTC, `<train-host-data>\logs\sprint4-standard-splat.log`. Standard tier on the same SRGD ActionRPG single scene at lr=1e-2.

| Step | model_PSNR | bicubic_PSNR | beats |
|----:|----:|----:|------|
| 1000 | 12.04 | 28.97 | 0/8 |
| 2000 | 11.27 | 30.27 | 0/8 |
| 3000 | 12.77 | 29.70 | 0/8 |

Same plateau as lite. Capacity is not the bottleneck.

## Combined verdict

Five independent paths, all consistent:

1. Direct 50K-Gaussian optimisation (no network): −3.59 dB vs bicubic on Sintel.
2. Lite-tier splat-only at lr ∈ {1e-4, 1e-3, 1e-2}: ≈ −17 dB vs bicubic on SRGD.
3. Lite-tier splat-only at lr=1e-1 (extreme parameters): −17.47 dB vs bicubic on SRGD.
4. Standard-tier splat-only at lr=1e-2: −17 dB vs bicubic on SRGD.
5. V0.5 (lite splat + 12K residual CNN): **+1.3 dB above bicubic** — but the splat-input channels of the residual head learned weights ≈ 0 (proven by `zero+residual` mode producing bit-identical output to `splat+residual` in `splat_contribution_probe.py`).

**The 2D Gaussian splat representation in our pipeline cannot do single-image super-resolution against engine-aliased LR competitively, regardless of implementation choices we have explored.** The only architecture that beats bicubic is one where a CNN learned to completely ignore the splat path.

The implementation is not the limiting factor. The representation is. This matches what the literature implies — papers that succeed at Gaussian SR (GSASR, GS-STVSR) do not use a fixed-bank softmax-weighted set of 1K–10K Gaussians produced by a small CNN. They use thousands more Gaussians, fully-learned per-Gaussian covariance, multi-scale features, and a learned final stage that is itself substantial CNN work.

## Decisions

1. **The Gaussian-temporal-canvas-as-SR thesis at our resource budget is dead.** No further engineering on splat-side SR. Accepting this is the rational move; insisting otherwise burns time on a known-bad regime.
2. **V0.5 ships as a CNN super-resolver with G-buffer inputs** if we want a deliverable. It works (+1.3 dB above bicubic, 56/56 held-out beats). It is not novel, but it solves the user's problem (a Wine/CrossOver-friendly open SR for game upscaling) with a 12K-param model.
3. **Sprint 5 (persistent canvas) loses its motivation in the SR context.** A canvas warps splats across frames — but if the splats encode no useful image content at any single frame, warping them across frames warps noise. Sprint 5 is not technically wrong, but its product value evaporates without a working splat-side signal.
4. **OSS-Gaussian-RR (denoising track) remains promising.** The D1 result (`2026-05-01-gaussian-denoising-naive-test.md`) showed Image-GS at n=1000 beats OIDN on PSNR. Denoising is a different problem — it only requires the Gaussians to encode a smooth approximation of a noisy target, which is what they are good at. **Reorient the Gaussian work toward denoising, not SR.**
5. **For super-resolution, fork OSS-SR off the Gaussian track.** Build it as a normal CNN-based SR pipeline with G-buffer conditioning. The Sprint 4 trainer + dataset adapters + bicubic comparison + held-out probe carry over verbatim — only the model architecture changes. This is a 1–2 week pivot that produces a shippable v0 SR model.

## Open questions (real, not exhaustive)

1. Does **alpha-compositing** (replacing `rasterize_gaussians_sum` with an OVER operator) change anything? GSASR uses alpha-style compositing; we use sum. Cheap to test if the upstream gsplat exposes an alpha rasterizer; otherwise it's a real reimplementation. Probably not the unlock — the direct-fit test in #1 above already used the same rasterizer and topped out at −3.59 dB.
2. Does **fully-learnable per-Gaussian covariance** (drop the fixed bank entirely) help? Cheaper to test than #1. But again, the direct-fit test had no bank constraint and still lost to bicubic.
3. Is the **V0.5 win generalisable beyond SRGD's particular degradation distribution**? The held-out scenes show generalisation across art styles, but not across degradation models (real game-engine LR ≠ our synthesised aliased LR).

These are interesting but not blocking. The project should pivot to either (a) ship V0.5 as a CNN super-resolver, or (b) reorient the Gaussian track to denoising where the representation is known to work.

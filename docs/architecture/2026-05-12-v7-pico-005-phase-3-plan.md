# v7-pico-005 — Phase 3 training-run plan

**Date:** 2026-05-12
**Goal:** First OSS-FX-validating training run. Train a HAT-Tiny + N-D-Gaussian-canvas teacher to ~30 dB PSNR @ α=1 (SR) and a measurable α=0.5 intermediate-frame metric on TartanAir-subsampled. ~100K steps, ~6 d on 3080 Ti.

This is the v7 analog of v6.2-pico-002 / v6.3-pico-003: a pico-tier **teacher** training run. The CNN student distillation comes after this run's metrics validate the OSS-FX hypothesis.

## Inputs

| Item | Value |
|---|---|
| Run id | `srcnn-v7.0-pico-005` |
| Architecture | v7, `backbone_kind="hat_tiny"`, enable_spawner=True |
| Backbone params | ~1.5M (HAT-Tiny transformer) |
| Spawner params | ~10K (tile decoder) |
| Composite head params | ~5K |
| Total trainable | ~1.5M |
| Dataset | TartanAir-subsampled (i, i+1, i+2) triplets via `TartanAirIntermediateTriplets` |
| Optimizer | AdamW, lr=2e-4, wd=1e-4, grad-clip max_norm=5.0 |
| Batch size | 2 (B=1 inner loop × accum=2 if needed for stability) |
| Resolution | 480×270 LR → 960×540 HR (scale=2) |
| Total steps | 100,000 |
| Checkpoint cadence | every 10K steps |
| Eval cadence | every 5K steps on a fixed held-out trajectory subset |

## α-curriculum (per spec)

Loss-weight ramp matches the v7 spec's prescription. The intermediate-frame supervision is unstable when the SR head is untrained, so we hold it back:

| Stage | Step range | `lambda_charbonnier` | `lambda_lpips` | `lambda_fg` | `lambda_fg_lpips` | `lambda_temp_consistency` | Notes |
|---|---|---|---|---|---|---|---|
| 1 (pure SR) | 0 – 20,000 | 1.0 | 1.0 | **0.0** | 0.0 | 0.0 | Backbone + spawner + head all converge on the α=1 task before FG enters the gradient. |
| 2 (add α=0.5) | 20,001 – 60,000 | 1.0 | 1.0 | linear-ramp 0.0→1.0 over 5K | 0.0 | 0.0 | FG aux ramps from 0 to its target weight to avoid loss-balance shock. |
| 3 (full OSS-FX) | 60,001 – 100,000 | 1.0 | 1.0 | 1.0 | 0.5 | 0.1 | LPIPS-VGG on the intermediate frame and temporal-consistency term both enabled. |

α=0.25 / α=0.75 supervision (deeper into the spec) is **deferred** to Phase 3.b. The current `TartanAirIntermediateTriplets` only exposes α=0.5; α=0.25 / 0.75 needs a (i, i+3) variant.

## Ablation matrix (compute budget: 3–5 ablations, $300–600 spot)

| Ablation | Δ vs baseline | Hypothesis |
|---|---|---|
| **A0 — baseline** | v7-pico-005 as specified above (curriculum + parent-child + no RRM, no Sobel, no Mip filters tweak) | Reference: OSS-FX teacher works at all |
| **A1 — canvas off** | Add `--enable-spawner-noop` (or equivalent: drop `--enable-parent-child` and zero `spawner_k_per_tile`) | Quantify the canvas contribution. If A1 ≈ A0 on α=1 metrics, the canvas is dead weight at the pico tier. |
| **A2 — parent-child off** | Drop `--enable-parent-child` from A0. Spawner still produces parents per tile; just no loss-adaptive density growth. | Tests whether parent-child carries its weight at this scale. Compares "uniform-density-only" vs "uniform + adaptive". |
| **A3 — no α-curriculum** | Drop `--curriculum` from A0; lambda_fg and lambda_temp ramp in from step 0 | Validates the curriculum's necessity. If A3 trains stably, the curriculum is over-cautious. |
| **A4 — placeholder backbone instead of HAT-Tiny** | Replace `--backbone-kind hat_tiny` with `--backbone-kind placeholder` | Cheap-baseline reference for the backbone choice. Expect noticeably worse α=1 metrics; if not, HAT-Tiny isn't pulling its weight at this size. |
| **B0 — Sobel HF on** | A0 plus `--lambda-sobel 0.1` | Tests whether the Sobel high-frequency edge loss improves thin-geometry preservation enough to justify its mild added noise sensitivity. |
| **B1 — RRM on** | A0 plus `--enable-rrm --rrm-loss-weight 2.0` | Tests whether STSS-style synthetic disocclusion augmentation improves real-disocclusion robustness. Measured via α=0.5 OSS-FX PSNR specifically, not α=1 SR PSNR. |

Run order: **A0 first (full 100K)**, then A1, A2, A3 to 60K to test ablation deltas, then B0 and B1 to 60K to test the new training signals. A4 is lowest priority. Each non-A0 ablation = ~3 days of 3080 Ti time.

## Metrics

Tracked every 5K steps on a held-out subset:

- **α=1 (SR)**: PSNR, SSIM, LPIPS-VGG vs frame N+1 GT. Compare directly against v6.2-pico-002 final numbers.
- **α=0.5 (OSS-FX)**: PSNR, SSIM, LPIPS-VGG vs frame N-half GT (frame i+1 in the subsampled triplet). This is the metric that did not exist for v6.x. Floor target: any value better than naive bicubic of the (i, i+2) midpoint.
- **Canvas health**: `count` over time; `count` post-prune; mean opacity; mean L_diag of the Cholesky packs (proxy for σ blowup).
- **Per-component loss curves**: `sr_charbonnier`, `sr_lpips`, `fg_charbonnier`, `fg_lpips`, `temp_consistency`, `total`. Logged each step to `history.jsonl` (already wired in `sr_train_v7.py`).

Pass criteria for the OSS-FX hypothesis (used to decide whether to start v7-pico-005-b distillation):

| Criterion | Threshold |
|---|---|
| α=1 PSNR | ≥ v6.2-pico-002 final – 0.5 dB (teacher class must not regress on the SR job) |
| α=0.5 PSNR | ≥ bicubic-midpoint baseline + 1.0 dB (OSS-FX has to do *something*) |
| Canvas count at end of training | 200–2,000 actives per frame (sanity — neither dead nor exploded) |
| Loss curves | Smooth, no NaN, no divergence after step 5K |

## Compute estimate

Per `2026-05-12-v7-nd-gaussians-spec.md` table: pico-tier single run ≈ 70–110 H100-hours equivalent. On 3080 Ti, step time will be ~4.5–7.0 s (1.5–2× v6.2 baseline due to N-D overhead), so 100K steps = 5–8 days wall-clock. Plan for **6 days** with checkpoint resumption if interrupted.

Ablations A1–A4 add another 3–5 days each at half-steps (60K), so 9–15 d of additional GPU time. Spot cost target on H100 if we migrate later: $600–$1K for the full ablation set.

## Host wiring (3080ti-windows)

Pre-flight checklist (run before kicking the job off):

1. `cd E:\oss-gaussian` → `git fetch && git checkout main && git pull` (resolve 472-file working-tree state with the user first — DO NOT clobber).
2. `conda activate image-gs` → `python -c "from oss.sr.v7.model import V7Model; print('ok')"` (verifies the v7 stack imports cleanly with the remote's torch).
3. Smoke test — use the dedicated launcher `scripts/3080ti/launch-v7-smoke.ps1` (commit e0cc578) which wraps the trainer with built-in pass/fail validation, OR invoke the trainer directly:

   ```bash
   python scripts/sr_train_v7.py \
       --tartanair-root E:/datasets/tartanair_extracted \
       --output-dir E:/checkpoints/srcnn-v7.0-smoke \
       --steps 50 --batch-size 2 --device cuda \
       --log-every 10 --max-triplets 8 \
       --backbone-kind hat_tiny \
       --curriculum \
       --enable-parent-child --parent-child-drift-rate 0.05 \
       --canvas-capacity 16384
   ```

   Expected: ~3 min runtime, history.jsonl has 5 entries, validation in launcher passes (canvas count grows, lambdas stay in stage-1 = 0 for fg, finite losses).

4. If smoke clean, kick the full job under WMI orphan-spawn (per `tailnet_3080ti.md`):

   ```bash
   python scripts/sr_train_v7.py \
       --tartanair-root E:/datasets/tartanair_extracted \
       --output-dir E:/checkpoints/srcnn-v7.0-pico-005 \
       --steps 100000 --batch-size 2 --device cuda \
       --log-every 100 --ckpt-every 10000 \
       --backbone-kind hat_tiny \
       --curriculum \
       --enable-parent-child --parent-child-drift-rate 0.05 \
       --canvas-capacity 16384 \
       --lambda-sobel 0.0    # OFF for first run (clean ablation; turn on for run b)
   ```

   `--enable-rrm` and `--lambda-sobel 0.1` are reserved for ablation runs A2+B (see § "Ablation matrix"). The CLI exposes them; the first scientific run keeps them off.

5. Register the v7 eval supervisor on the 3080 Ti by running `scripts/3080ti/register-v7-supervisors.ps1` (commit 6097e2f). Once registered, the supervisor will auto-eval every new checkpoint via `sr_eval_v7.py` and append to `score_log_v7.json`, which the v7 dashboard lane (build_public_dashboard.py with `--version v7`) surfaces.

## Risks / open items

- **B=1 inner loop**: at batch_size=2 the trainer iterates samples sequentially through the per-rank canvas. Step time roughly doubles vs. a true batched forward. Acceptable at 3080 Ti; revisit if we move to multi-GPU.
- **Dataset triplet count**: `TartanAirIntermediateTriplets` filters cross-trajectory pairs. Real triplet count after filtering is unknown until the smoke test runs; if it's below ~20K, consider supplementing with Vimeo-90K (already on the spec's data list).
- **Held-out player rendering**: dashboard wiring for v7 frames may need a parallel update — the v6.x held-out frame plumbing assumes 2D outputs and `t_query` is a new axis we haven't dashboarded yet. Deferred until v7-pico-005 actually emits a checkpoint worth visualizing.
- **GAN head**: spec says "+ GAN, as v6.2" for SR loss. The trainer currently has no GAN; this is OK for the first run and an obvious add for v7-pico-005-b distillation.

## What this does NOT include

- Distillation to the ≤0.4M CNN student. That's the Phase 4 deliverable; v7-pico-005 is just the teacher.
- CUDA-kernel port of the N-D rasterizer. The pure-Python ref is fine for 3080 Ti at this scale; the CUDA detour decision (`project_cuda_kernel_decision.md`) is parallel.
- Cross-engine fine-tune on captured game footage. Phase 5.

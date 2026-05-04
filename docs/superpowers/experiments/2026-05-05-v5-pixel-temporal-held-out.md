# 2026-05-05 — v5-pixel-temporal held-out vs v4 (TartanAir, manifest-pinned)

**Status:** [PENDING — fill in after morning eval]
**Question:** Does v5-pixel-temporal meet the success criteria from the design spec?
**Spec:** `docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md` §Success criteria

## Context

Trained 2026-05-04 18:07 → ~22:50 CDT on `<train-host>` (RTX 3080 Ti, Ampere, 12 GB).

- **Warm-start:** `srcnn-prod-v4-lpips/step-00385000.pt` (sha256 `8C079615…`)
- **Final ckpt:** `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/step-00080000.pt`
- **Training data:** TartanAir Easy (Sintel deferred — Sintel Depth was downloaded mid-run; Phase 3 fell back to TartanAir per the launch-status note)
- **Loss schedule:** Phase 1 (0–10K, backbone frozen, L1+SSIM); Phase 2 (10K–60K, full unfreeze, L1+SSIM+LPIPS+temporal-consistency); Phase 3 (60K–80K, LR×0.01 fine-tune)

## Method

Manifest-pinned deterministic eval per Codex C9 (commit `a472851`) using `<train-host-data>/checkpoints/v5_held_out_manifest.json` (64 frame pairs from TartanAir held-out trajectories, seed=0, lr_scale=2.0, lr_synth_args fixed).

```powershell
ssh <train-host>
cd <train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_temporal_held_out.py `
    --ckpt-temporal <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\step-00080000.pt `
    --ckpt-baseline <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt `
    --tartanair-root <train-host-data>\datasets\tartanair_extracted `
    --manifest <train-host-data>\checkpoints\v5_held_out_manifest.json `
    --n-samples 64
```

Output written to `held_out_results.json` next to the temporal checkpoint.

## Results

PASTE the final block of the script output here. Format mirrors `scripts/sr_v3_vs_v4_ab.py`:

```text
=== A/B fixed-batch eval (TartanAir manifest, n=64) ===
  ckpt_a = step-00385000.pt   (v4, L1+SSIM+0.1·LPIPS)
  ckpt_b = step-00080000.pt   (v5-pixel-temporal)

PSNR (dB, higher is better)
  A          : XX.XXX
  B          : XX.XXX
  bicubic    : XX.XXX
  B-vs-A     : +X.XXX dB
  A>bicubic  : XX/64
  B>bicubic  : XX/64
  B>A        : XX/64

LPIPS-VGG (lower is better)
  A          : 0.XXXX
  B          : 0.XXXX
  bicubic    : 0.XXXX
  B-vs-A     : -0.XXXX  (-XX.X%)
  A<bicubic  : XX/64
  B<bicubic  : XX/64
  B<A        : XX/64

Temporal stability — mean(|warp(out_t, motion_t→t+1) - out_{t+1}|_1)
  v5-temporal: X.XXXX
  v4-baseline: X.XXXX
  ratio      : X.XX (lower-better; <1 = v5 wins on stability)
```

## Success criteria (from spec)

- [ ] PSNR ≥ +1.5 dB over v4 baseline
- [ ] LPIPS ≤ 0.20 (vs v4 ~0.31)
- [ ] Temporal stability: warp-then-diff between t and t+1 ≤ 0.5× the v4 single-frame variance
- [ ] No regression on bicubic-beats: ≥ 95% of held-out frames beat bicubic on both PSNR AND LPIPS

## Conclusion

[PENDING — fill in one of:]

- **PASS — ship v5-pixel-temporal.** All four success-criteria boxes checked. Update `README.md` S5 row with the held-out numbers. Move on to Gaussian training (sequential GPU per Cash's directive). Loser of the v5 race becomes v6+ research input per the spec.
- **PARTIAL PASS — ship v5-pixel-temporal with caveats.** Three of four criteria met; document which one missed and by how much. Decide whether the miss is small enough to ship anyway OR whether to run a Sintel-fine-tune follow-up (see `docs/superpowers/notes/2026-05-04-claude-codex-asks-r4.md` C14) before declaring v5.
- **FAIL — do not ship v5-pixel-temporal.** Fewer than three criteria met. Keep v4 as production single-frame; pixel-temporal becomes a research artifact. Continue with Gaussian race anyway (it might still win).

## Caveats / honest limits

- Single-batch eval on a TartanAir held-out trajectory; rerun on Sintel after Sintel Depth integration is fully wired (see C13 + C14).
- LPIPS-VGG is one perceptual metric with known biases. Other metrics (DISTS, FID-on-game-content, human raters) are not used here — same caveat as the v3-vs-v4 A/B.
- Cold-start regime in the eval: `prev_hr_t = bicubic(LR_t)`, `prev_hr_{t+1} = baseline_output_at_t.detach()`. This matches the deployed inference engine's first-frame behavior. Recurrent rollout (passing the temporal output back as prev_hr) would test the model's longer-horizon stability — that's a separate experiment.

## Follow-ups (regardless of outcome)

1. ONNX export via `scripts/sr_export_temporal_onnx.py` (Codex C10, commit `31aad5b`) on the v5 ckpt. Catches opset-17 / dynamic-axis issues early.
2. TRT FP16 export with the narrow profile pattern from `scripts/sr_export_tensorrt.py`.
3. (If pass) launch v5-gaussian-temporal training per `docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`.
4. (If partial/fail) post-mortem memo: which design choice contributed most to the gap? (warp sign convention? LPIPS weight? temporal-consistency weight? backbone freeze duration?)

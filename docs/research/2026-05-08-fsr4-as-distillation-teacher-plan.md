# FSR 4 as Distillation Teacher — v6.2.1 Plan

**Date:** 2026-05-08
**Status:** plan / pre-implementation
**Parent:** `docs/architecture/2026-05-08-v62-arch-v4-spec.md`
**Companion:** `docs/research/2026-05-08-fsr4-architecture-observations.md`
**Tracking issue:** GitHub #7

---

## TL;DR

After v6.2-pico-002 baseline converges, retrain in v6.2-pico-002.1 with FSR 4 as a **distillation teacher** alongside HAT-Tiny. FSR 4 was MIT-licensed in AMD's SDK 2.0.0 (orphan commit `01446e6a`, force-pushed away later but the irrevocable MIT grant remains). This is a quality unlock no other SR project can replicate — STSS had to reverse-engineer DLSS quality from inputs/outputs alone; we get a queryable competitive teacher whose outputs we can train against directly.

---

## Why this works

1. **MIT-licensed source** lets us run FSR 4 binary in our training pipeline, capture outputs, train against them — no copyright issues.
2. **FSR 4 is a strong reference quality target** — public AMD claims of ~30+ dB PSNR on Cyberpunk 2077 / Hitman 3 at 1440p→4K. Even with our held-out batch being TartanAir oldtown (different content), FSR 4 outputs on the same input produce a high-quality target.
3. **Architecture-orthogonal teacher** — FSR 4 is conv+FasterNet-based, OSS is canvas-residual. Training OSS to match FSR 4 outputs preserves OSS's architectural advantages while inheriting FSR 4's quality.
4. **Reproducible teacher** — unlike running a teacher network in pure inference (which can vary across hardware), FSR 4 binary is a fixed signed DLL. Outputs are deterministic given inputs; training consistency improves.

---

## Pipeline architecture

```
Training step n in v6.2-pico-002.1:
  Input:  LR + G-buffer (depth, motion, normals)
  Output: HR (predicted by OSS)

  Loss = L_appearance + λ_distill_hat * L_distill_hat + λ_distill_fsr4 * L_distill_fsr4

Where:
  L_appearance = L1(OSS_HR, GT_HR) + 0.1 * LPIPS(OSS_HR, GT_HR) + 0.1 * SSIM(...)
  L_distill_hat = L1(OSS_HR, HAT_teacher(LR + G-buffer))
  L_distill_fsr4 = L1(OSS_HR, FSR4_teacher(LR + G-buffer + jitter))
```

Both teachers run forward-only; gradients flow only through the OSS student path.

---

## FSR 4 teacher integration

### Input compatibility

FSR 4's `ffx_provider_fsr4_dx12.cpp` host code expects:

- **Color input** (LR rendered frame, linear-light)
- **Depth** (from G-buffer)
- **Motion vectors** (from G-buffer)
- **Jitter offset** (sub-pixel, sampled from a Halton-like sequence per frame)
- **Output target** (HR resource for the upscaled frame)
- **Frame time delta** (used for temporal accumulation)

OSS already uses TartanAir + UE5 captures with depth + MV + jitter. The capture layer (in `oss/sr/v6/dataset.py` and the UE5 capture pipeline) needs an FSR 4-compatible export path. Specifically:

1. Color must be **linear-light** (FSR 4 has a non-linear flag but linear is the recommended path).
2. Depth must use the **inverse-Z** convention FSR 4 expects (shipping games use this; might need a conversion if our captures use a different sign convention).
3. Motion vectors in **HR pixel units**, OR in LR pixel units with the FSR 4 motion-scale parameter set to the upscale ratio.
4. Jitter sequence must be the same one OSS used during the training-corpus capture (otherwise FSR 4 sees wrong jitter and produces lower-quality output).

### Runtime architecture

**Phase 1 — Pre-compute teacher outputs (one-time, before training starts):**

```python
# scripts/precompute_fsr4_teacher.py
# Run FSR 4 binary on the entire training corpus once, write the HR outputs to
# a parallel directory. Avoids running FSR 4 inference inside the training loop
# (deterministic + decouples the teacher availability from training infra).

for capture in training_corpus:
    lr, depth, mv, jitter = capture.inputs
    fsr4_hr = run_fsr4(lr, depth, mv, jitter, output_size=lr.size * 2)
    save(fsr4_hr, capture.fsr4_teacher_path)
```

This requires Windows + DX12 + the FSR 4 binary loaded via the FidelityFX API. Run on the 3080 Ti (Windows host) or any RDNA 2+ GPU (FSR 4 has driver-level support starting RDNA 4 native, with the AMD-FSR-4-INT8 README's driver-DLL workaround for RDNA 2/3).

Storage: per HR sample at 1440p RGBA fp16 ≈ 24 MB. For 100K samples ≈ 2.4 TB. Compress to **bf16 + lossless PNG** (~3-5 MB per sample) → ~300-500 GB. Manageable on a dev rig with TBs of disk; if it's too large, randomly sample 20% of the corpus to teach against (industry standard for distillation).

**Phase 2 — Training loop reads pre-computed teacher outputs:**

```python
# Inside the v6.2 trainer:
for batch in dataloader:
    lr, depth, mv, gt_hr, fsr4_hr = batch  # fsr4_hr loaded from disk
    oss_hr = oss_model(lr, depth, mv)
    hat_hr = hat_teacher(lr, depth, mv).detach()  # frozen teacher

    L_appearance = l1_loss(oss_hr, gt_hr) + 0.1 * lpips(oss_hr, gt_hr)
    L_distill_hat = l1_loss(oss_hr, hat_hr)
    L_distill_fsr4 = l1_loss(oss_hr, fsr4_hr)

    L = L_appearance + lambda_hat * L_distill_hat + lambda_fsr4 * L_distill_fsr4
    L.backward()
    optimizer.step()
```

### Loss weighting schedule

Initial:

- `λ_distill_hat`: 0.3, decay to 0.05 over warmup (HAT-Tiny is the architecture-aligned teacher, but its quality ceiling is what we're trying to BEAT)
- `λ_distill_fsr4`: 0.5, hold for the first 50K steps then anneal to 0.2 (FSR 4 is the high-quality teacher; want stronger pull early, taper as OSS finds its own optimum that may exceed FSR 4 on some metrics)

Tunable; ablate `λ_distill_fsr4 ∈ {0, 0.2, 0.5, 0.8}` in early experiments.

### What to NOT do

- **Do not match FSR 4 architecturally.** OSS's differentiator is canvas-residual + structural FG. Distilling OUTPUTS preserves the architectural advantage; copying the architecture loses it.
- **Do not distill on FSR 4 frame-generated frames.** Frame-gen is a separate AMD pipeline that uses optical flow. We only distill on the SR (upscale) outputs. Frame-gen comparison should be against the canvas-warp path which is already structurally free.
- **Do not infer FSR 4 inside the training loop.** Pre-compute in Phase 1, read from disk in Phase 2. Keeps training infra independent of the FSR 4 runtime.

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FSR 4 binary requires RDNA 4 hardware not available to us | Medium | The AMD-FSR-4-INT8 README documents a driver-DLL swap for RDNA 2/3. Test on 3080 Ti — if NVIDIA hardware can't run it, use a Steam Deck (RDNA 2) loaner. |
| Pre-computing teacher outputs takes too long | Low-medium | If pre-compute is >1 day, reduce to 20% sample of corpus (standard distillation practice) |
| FSR 4 quality on TartanAir is poor (out-of-distribution for AMD's training data) | Medium | Quality won't match Cyberpunk 2077 numbers — TartanAir is synthetic + lower visual complexity. Still a useful teacher relative to OSS-baseline; ablate the `λ_distill_fsr4=0` case to verify it actually helps. |
| FSR 4 is biased toward AMD's training distribution and produces non-representative outputs | Medium | Same as above. Ground truth is GT_HR, not FSR 4. The distillation just provides a high-quality reference signal alongside the GT. |
| OSS's structural advantages (canvas + FG) compromise when forced to match FSR 4's per-frame conv outputs | Medium-low | Keep λ_distill_fsr4 modest (0.5 max) so the appearance loss + canvas-coherence regularizers retain dominance. Watch held-out PSNR/LPIPS at the structural-advantage-relevant scenes (long camera moves, disocclusions). |
| License compliance (MIT preservation in derived ckpts) | Low | Distilling against a teacher's outputs IS NOT considered a "derivative work of the teacher's source code" under MIT or copyright law. The trained weights are our own; we preserve the MIT attribution for the FSR 4 source we vendored. |

---

## Acceptance gates

- v6.2-pico-002 (no FSR 4 distillation) baseline trained to convergence on TartanAir oldtown held-out → publish numbers as the architecture-only baseline.
- v6.2-pico-002.1 (with FSR 4 distillation) retrained from same init on same corpus, same step count → measure `Δ_PSNR` and `Δ_LPIPS` vs. baseline.
- Pass: `Δ_PSNR ≥ +0.3 dB` AND `Δ_LPIPS ≤ -0.01` over baseline. Otherwise the distillation isn't carrying its weight; ablate.

---

## Implementation order (after pico-002 baseline ships)

1. `scripts/precompute_fsr4_teacher.py` — runs FSR 4 binary on a corpus subset (start with 20% sample; expand if quality gates pass). Writes `<corpus>/fsr4_teacher/<scene>/<frame>.png` next to ground truth.
2. `oss/sr/v6/dataset.py` — load FSR 4 teacher output as an additional batch field when present.
3. `oss/sr/v6/losses.py` — new `distill_fsr4_loss(oss_hr, fsr4_hr)` with the schedule.
4. `scripts/sr_train_v6.py` — add `--fsr4-teacher-dir` flag + the loss schedule wiring.
5. `configs/v6.2-pico-002.1.yaml` — derived from `v6.2-pico-002.yaml` with distillation enabled.

---

## Followups

- Investigate using the FSR 4 binary's `Quality` mode (1.5×) outputs as an additional teacher signal alongside `Performance` mode (2.0×) — different data points, may help generalization across upscale ratios.
- After pico-002.1 ships, consider running OSS as ITS OWN teacher in pico-002.2 — self-distillation has been shown to be effective for further refinement once a strong base is in place.
- If FSR 4 teacher outputs are stored on disk, they can also serve as a benchmark for the "FSR 4 measured" reference line in the dashboard's per-run held-out chart overlay (closing the loop on issue #6).

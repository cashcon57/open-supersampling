# v5 Gaussian Temporal — Train-Start Memo

**Status:** Pre-launch (lab-notebook discipline: this memo is committed BEFORE GPU time is burned)
**Date:** 2026-05-04
**Author:** Cash Conway + Claude (Opus 4.7)
**Sprint:** S5 (v5 dual-track — Gaussian temporal research track)
**Spec:** [`docs/superpowers/specs/2026-05-04-v5-gaussian-temporal-design.md`](../specs/2026-05-04-v5-gaussian-temporal-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-04-v5-gaussian-temporal-plan.md`](../plans/2026-05-04-v5-gaussian-temporal-plan.md)
**Runbook:** [`docs/superpowers/notes/2026-05-04-v5-gaussian-temporal-runbook.md`](../notes/2026-05-04-v5-gaussian-temporal-runbook.md)

---

## Hypothesis

Test the hypothesis that 2D Gaussians as a persistent temporal scene memory can outperform pixel-based temporal accumulation (the v5-pixel-temporal control track) for real-time super-resolution. The unique advantages we're testing:

- Analytical sub-pixel warping (no resample blur compounding across frames)
- Continuous representation with persistent positions (sub-pixel jitter accumulation is structural, not learned)
- Tractable token count for multi-frame attention (~5K Gaussians vs millions of pixels per frame)
- Densification under disocclusion (clean newly-visible region handling, no pixel-rejection blockiness)

This is **research-grade work**. No production deployment of Gaussian temporal SR exists. We are deliberately running this in parallel with the proven pixel track so that we have a safe fallback if this fails.

(Verbatim from spec §Goal.)

---

## Success criteria

The Gaussian track wins or ties the pixel track on the **same fixed held-out batch** used for the v3-vs-v4 A/B:

- [ ] PSNR ≥ pixel track − 0.3 dB (tie acceptable)
- [ ] LPIPS ≤ pixel track − 0.01 (genuine perceptual win)
- [ ] Temporal stability (warp-then-diff between t and t+1) ≤ pixel track variance
- [ ] Inference latency ≤ 1.5× pixel track at 1080p→4K on RTX 3080 Ti

If Gaussian beats pixel, we ship Gaussian as v5. If Gaussian ties or loses, we ship pixel as v5 and continue Gaussian as v6+ research.

(Verbatim from spec §Success criteria.)

---

## Training schedule

Four phases, ~140K total steps (verbatim from spec §Training §Schedule):

1. **Phase 1 — single-frame Gaussian fitter (steps 0–20K):** train only the per-frame fitter to produce Gaussians from one frame. No temporal. This gets the rasterizer + densification stable in isolation. Reuse V0.5 splat infrastructure.
2. **Phase 2 — temporal warp + transformer warmup (steps 20K–50K):** add prev-frame Gaussian warp + small transformer (2 layers). Frozen fitter. Establishes the temporal update head can learn.
3. **Phase 3 — joint training (steps 50K–120K):** unfreeze fitter, full transformer (4–6 layers), full loss including temporal consistency. Densification active.
4. **Phase 4 — Sintel fine-tune (steps 120K–140K):** real-data polish.

Sequence sampling: trajectory window of 5–7 consecutive frames per training step (vs the pixel track's 2-frame pairs); ~3× compute per step.

---

## Expected runtime

**24–48 hours on RTX 3080 Ti** (per spec §Training §Schedule). Multi-frame transformer attention plus per-step trajectory windows of 5–7 frames put per-step compute at roughly 3× the pixel track.

---

## Warm-start checkpoint

**None — cold-start run.** This is a research-grade architecture with no prior compatible weights. Phase 1 is explicitly designed as a single-frame fitter warm-up that reuses V0.5 splat infrastructure conceptually, but no `.pt` is loaded at launch. Reproducibility hash will be the v0.2-dev commit SHA recorded during pre-flight.

---

## Output checkpoint directory

`<train-host-data>/checkpoints/srcnn-v5-gaussian-temporal/`

Logs: `<train-host-data>/checkpoints/srcnn-v5-gaussian-temporal/train.log` (orphan-spawn redirect target).

---

## Datasets

- **TartanAir Easy (primary):** `<train-host-data>/datasets/tartanair_extracted/` — 18 environments, ~600 GB, sequential trajectories with real flow + depth.
- **Sintel clean pass (held-out + Phase 4 fine-tune):** `<train-host-data>/datasets/sintel/` — ~1041 frames, real Blender ground truth.

Sequence sampling: trajectory pick → start index `i` → frames `(i, i+1, …, i+W-1)` for window `W ∈ [5, 7]` at the same scene+camera state. Augmentation: random crop, horizontal flip (with motion-vec sign flip), brightness/contrast jitter matched across the entire window.

LR synthesis: `EngineAliasedLRSynth` (Halton jitter + TAA blur + JPEG q=85 + blur σ=1.5), applied identically to all frames in a window with consistent jitter offsets.

---

## GPU-share decision (CRITICAL)

The remote 3080 Ti is shared between the v5-pixel-temporal control track and this v5-gaussian-temporal research track. **Cash directive:** "Sequential GPU train unless overlap is safe; test overlap first."

**Default policy:** the Gaussian launch waits until either
1. The pixel-temporal training run has reached `step-00080000.pt` (final), OR
2. The pixel run has stopped writing new checkpoints and `nvidia-smi` shows the GPU idle / memory free,

before issuing the WMI orphan-spawn. The runbook pre-flight encodes both checks as gating commands. Overlap is opt-in and only after a smoke run with both processes resident shows no OOM or thrash.

---

## Launch command (PowerShell, WMI orphan-spawn)

This is the literal command line that will be invoked from the <train-host> host. Orphan-spawn pattern means SSH disconnects cannot kill the run.

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_gaussian_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal --tartanair-root <train-host-data>\datasets\tartanair_extracted --sintel-root <train-host-data>\datasets\sintel --max-steps 140000 > <train-host-data>\checkpoints\srcnn-v5-gaussian-temporal\train.log 2>&1'
}
```

See the runbook for full pre-flight + GPU-share gate + monitoring sequence.

---

## Post-run gate

After training completes, run the held-out eval per Plan Task 12 and fill in `docs/superpowers/experiments/2026-XX-XX-v5-gaussian-temporal-held-out.md`. The Sprint-5 closeout (Plan Task 14) compares pixel vs Gaussian on the same fixed held-out batch and executes the ship decision per the spec — Gaussian must explicitly beat pixel; tie ≠ win.

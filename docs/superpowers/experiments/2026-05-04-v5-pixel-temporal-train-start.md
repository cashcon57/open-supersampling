# v5 Pixel Temporal — Train-Start Memo

**Status:** Pre-launch (lab-notebook discipline: this memo is committed BEFORE GPU time is burned)
**Date:** 2026-05-04
**Author:** Cash Conway + Claude (Opus 4.7)
**Sprint:** S5 (v5 dual-track — pixel temporal control track)
**Spec:** [`docs/superpowers/specs/2026-05-04-v5-pixel-temporal-design.md`](../specs/2026-05-04-v5-pixel-temporal-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-04-v5-pixel-temporal-plan.md`](../plans/2026-05-04-v5-pixel-temporal-plan.md)
**Runbook:** [`docs/superpowers/notes/2026-05-04-v5-pixel-temporal-runbook.md`](../notes/2026-05-04-v5-pixel-temporal-runbook.md)

---

## Hypothesis

Add FSR 2-class temporal accumulation to the v4 single-frame SR-CNN. Closes the perceptual quality gap to FSR 2 / DLSS 2 by reusing motion vectors and prior-frame HR output. Serves as the **control track** in the v5 dual-track experiment — the proven recipe against which the experimental Gaussian-temporal track is measured.

(Verbatim from spec §Goal.)

---

## Success criteria

Held-out fixed-batch eval on Sintel + a TartanAir held-out trajectory (not seen in training):

- [ ] PSNR ≥ +1.5 dB over v4 baseline
- [ ] LPIPS ≤ 0.20 (vs v4 ~0.31)
- [ ] Temporal stability: warp-then-diff between frames t and t+1 ≤ 0.5× the v4 single-frame variance
- [ ] No regression on bicubic-beats: ≥ 95% of held-out frames beat bicubic on both PSNR AND LPIPS

(Verbatim from spec §Success criteria.)

---

## Training schedule

Three phases, ~80K total steps (verbatim from spec §Schedule):

1. **Phase 1 — warm-up (steps 0–10K):** freeze v4 backbone, train only the temporal head + disocclusion params. Loss: appearance only (no temporal consistency yet). Establishes that the head can learn the warp+blend pattern.
2. **Phase 2 — joint (steps 10K–60K):** unfreeze backbone with reduced LR (10% of head LR). Add temporal consistency loss. Full loss active.
3. **Phase 3 — fine-tune on Sintel (steps 60K–80K):** small LR (1% of base), Sintel-only. Real-data polish.

---

## Expected runtime

**12–16 hours on RTX 3080 Ti** (per spec §Schedule). The temporal consistency loss requires sampling two consecutive frames per training step and rendering both — ~1.5× compute per step vs v4.

---

## Warm-start checkpoint

- **Path (remote):** `<train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt`
- **SHA256:** `8C079615E6ED2580E21615AB677F16C9B646FB00B74C507617F70B1F6691BEF9`
- **Source:** v4 production LPIPS run (Sprint 4).
- **Verification command (re-run on remote before launch):**
  ```powershell
  Get-FileHash <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt -Algorithm SHA256
  ```

---

## Output checkpoint directory

`<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/`

Logs: `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal/train.log` (orphan-spawn redirect target).

---

## Datasets

- **TartanAir Easy (primary):** `<train-host-data>/datasets/tartanair_extracted/` — 18 environments, ~600 GB, sequential trajectories with real flow + depth.
- **Sintel clean pass (held-out + Phase 3 fine-tune):** `<train-host-data>/datasets/sintel/` — ~1041 frames, real Blender ground truth.

Sequence sampling: trajectory pick → start index `i` → frames `(i, i+1)` at the same scene+camera state. Augmentation: random crop, horizontal flip (with motion-vec sign flip), brightness/contrast jitter matched across the pair.

LR synthesis: `EngineAliasedLRSynth` (Halton jitter + TAA blur + JPEG q=85 + blur σ=1.5), applied identically to both frames in a pair with consistent jitter offsets.

---

## Launch command (PowerShell, WMI orphan-spawn)

This is the literal command line that will be invoked from the <train-host> host. Orphan-spawn pattern means SSH disconnects cannot kill the run.

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine='cmd /c cd /d <train-host-data>\oss-gaussian && <windows-home>\Miniconda3\envs\image-gs\python.exe scripts\sr_train_temporal.py --output-dir <train-host-data>\checkpoints\srcnn-v5-pixel-temporal --warm-start <train-host-data>\checkpoints\srcnn-prod-v4-lpips\step-00385000.pt --tartanair-root <train-host-data>\datasets\tartanair_extracted --sintel-root <train-host-data>\datasets\sintel --max-steps 80000 > <train-host-data>\checkpoints\srcnn-v5-pixel-temporal\train.log 2>&1'
}
```

See the runbook for full pre-flight + monitoring sequence.

---

## Post-run gate

After training completes, run the held-out eval per Plan Task 10 and fill in `docs/superpowers/experiments/2026-XX-XX-v5-pixel-temporal-held-out.md`. README S5 row updates only if all four success-criteria boxes pass.

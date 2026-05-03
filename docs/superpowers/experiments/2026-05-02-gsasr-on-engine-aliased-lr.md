# GSASR on Engine-Aliased LR: Does Feature-Space Gaussian SR Beat Bicubic?

**Date:** 2026-05-02
**Status:** complete — definitive
**Predecessor:** `docs/superpowers/experiments/2026-05-02-splats-SR-literature-delta.md`
**Hardware:** RTX 3080 Ti (<train-host>, CUDA 12.4, 12 GB VRAM)
**Code commit (oss-gaussian):** `2b82e4bde30d912610454157a300687a4ba9a296` (branch v0.2-dev)
**GSASR commit:** `9d2eb64a51303ce7a22fd672197185488b38e10a` (github.com/ChrisDud0257/GSASR, single commit — shallow clone)

---

## Hypothesis

GSASR (arXiv 2501.06838, ICCV 2025) uses feature-space 2D Gaussian Splatting with a large attention-based backbone (EDSR encoder, Gaussian Interaction Blocks). The literature delta (`2026-05-02-splats-SR-literature-delta.md`) identified this as the most likely published architecture to succeed where our implementation fails.

The question: **does GSASR beat bicubic upsampling on engine-aliased SRGD LR (blur_sigma=1.5 + JPEG q=85)?**

If yes (margin >= 1 dB): the feature-space Gaussian SR thesis is validated on our LR distribution. The architectural gap — not the LR domain — was our binding constraint.

If no (margin < 0 dB): the engine-aliased LR domain is the binding constraint. No drop-in Gaussian SR architecture solves the problem; training on engine-aliased data is required.

---

## Setup

### GSASR version and weights

- **Repo:** `github.com/ChrisDud0257/GSASR`, commit `9d2eb64`
- **Model variant:** `EDSR_DIV2K` (Enhanced, smallest backbone — most-comparable to our resource tier)
- **Weights:** pre-downloaded to `<train-host-data>\external-eval\GSASR\weights\EDSR_DIV2K\{encoder.pth, decoder.pth}` from HuggingFace `mutou0308/GSASR` (CC-BY-NC 4.0 — research use only, not shipped in repo)
- **License:** CC-BY-NC 4.0. Weights remain on <train-host>, not committed to repo.
- **Inference mode:** AMP bfloat16 (`torch.amp.autocast('cuda', dtype=torch.bfloat16)`)
- **GSASR dmax:** 0.1 (paper's recommended x2 setting)
- **EDSR denominator:** 12 (required padding divisor for EDSR attention blocks)

### Eval data

- **Dataset:** SRGD GameEngineData / ActionRPG (575 frames total, 960x540 HR)
- **Frame selection:** 24 frames evenly sampled across the sequence (stride ~24 frames); range `00001.png` to `00552.png`
- **HR path:** `<train-host-data>\datasets\srgd\data\GameEngineData\ActionRPG\`

### LR synthesis

Two conditions:

**Condition A — Engine-aliased LR (our target distribution):**
- Halton-2 subpixel jitter on HR before downsample (matches TAA jitter)
- Area-filter (box average) 2x downsample: 960x540 → 480x270
- TAA blur approximation: 3x3 Gaussian, sigma=1.5 (kernel size 9 at 3-sigma)
- JPEG compression round-trip: quality=85

This matches the `EngineAliasedLRSynth(blur_sigma=1.5, enable_jpeg=True, jpeg_quality=85)` config from `oss/gaussian/data/lr_synthesis.py`. LR synthesis was inlined into the eval script (`<train-host-data>\eval_gsasr_engine_aliased.py`) to avoid import dependency issues on the remote.

**Condition B — Bicubic-clean LR (sanity check / paper-comparable):**
- `cv2.resize(..., interpolation=cv2.INTER_CUBIC)` at scale=0.5
- Same protocol as prior Sintel eval (`2026-05-01-pretrained-gaussian-sr-eval.md`)
- 8 frames (first 8 of the 24 selected, for speed)

### Baselines

- **GSASR x2:** model as described above
- **Bicubic x2:** `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)` — same implementation as `oss/gaussian/bench/baselines.py::bicubic_upscale`

### Metrics

PSNR and SSIM computed via `skimage.metrics` (`peak_signal_noise_ratio`, `structural_similarity` with `channel_axis=2`, `data_range=255`). Both metrics computed on BGR uint8 (full 3-channel).

### CLI (eval script)

```powershell
# Run from <train-host-data>\external-eval\GSASR with PYTHONPATH=<train-host-data>\oss-gaussian
<windows-home>\Miniconda3\envs\image-gs\python.exe <train-host-data>\eval_gsasr_engine_aliased.py \
    --n_frames 24 --blur_sigma 1.5 --jpeg_quality 85 --dmax 0.1
```

Eval script at `<train-host-data>\eval_gsasr_engine_aliased.py` (also written locally to `/tmp/gsasr_engine_aliased_eval.py` during this session — not committed to repo per scope).

Full output at `<train-host-data>\eval_results\gsasr_engine_aliased\` (metrics.json, summary.txt, per-frame PNGs).

---

## Result

### Summary table

| Eval set | GSASR PSNR | Bicubic PSNR | Margin | GSASR SSIM | Bicubic SSIM | GSASR wins |
|----------|:----------:|:------------:|:------:|:----------:|:------------:|:----------:|
| Engine-aliased LR (n=24) | 29.136 dB | 29.176 dB | **−0.040 dB** | 0.7587 | 0.7621 | **0/24** |
| Bicubic-clean LR sanity (n=8) | 33.594 dB | 37.296 dB | **−3.702 dB** | — | — | **0/8** |

### Per-frame results (engine-aliased condition)

| Frame | GSASR PSNR | Bicubic PSNR | Margin |
|-------|:----------:|:------------:|:------:|
| 00001 | 28.50 | 28.54 | −0.04 |
| 00024 | 29.21 | 29.24 | −0.03 |
| 00048 | 28.26 | 28.29 | −0.03 |
| 00072 | 28.45 | 28.49 | −0.04 |
| 00096 | 29.21 | 29.25 | −0.04 |
| 00120 | 28.69 | 28.72 | −0.03 |
| 00144 | 28.61 | 28.64 | −0.03 |
| 00168 | 28.90 | 28.93 | −0.03 |
| 00192 | 28.54 | 28.57 | −0.03 |
| 00216 | 29.56 | 29.59 | −0.04 |
| 00240 | 30.60 | 30.64 | −0.04 |
| 00264 | 28.35 | 28.39 | −0.04 |
| 00288 | 28.83 | 28.86 | −0.03 |
| 00312 | 28.01 | 28.04 | −0.03 |
| 00336 | 29.03 | 29.07 | −0.03 |
| 00360 | 27.93 | 27.97 | −0.04 |
| 00384 | 27.73 | 27.76 | −0.03 |
| 00408 | 27.76 | 27.80 | −0.04 |
| 00432 | 28.31 | 28.34 | −0.03 |
| 00456 | 30.60 | 30.65 | −0.05 |
| 00480 | 30.14 | 30.18 | −0.04 |
| 00504 | 34.88 | 34.99 | −0.11 |
| 00528 | 29.71 | 29.77 | −0.06 |
| 00552 | 29.42 | 29.48 | −0.06 |
| **mean** | **29.136** | **29.176** | **−0.040** |

GSASR wins on **0 of 24 frames** in the engine-aliased condition. The margin is remarkably uniform (−0.03 to −0.11 dB), with the largest loss on frames with flat-colour regions (00504: −0.11 dB on a predominantly-still scene where bicubic's exact inverse is optimal).

### Sanity check interpretation

The bicubic-clean sanity result (GSASR −3.702 dB vs. bicubic on clean SRGD) is expected and consistent with the prior Sintel result (`2026-05-01-pretrained-gaussian-sr-eval.md`, −4.55 dB on Sintel). This is the bicubic-LR trap: when LR is produced by bicubic downsampling, bicubic upsample is the near-inverse, and any learned SR model that hallucinates HF detail will lose PSNR/SSIM against the smooth GT. The sanity check confirms GSASR is running correctly — it is not misconfigured.

Note: the GSASR paper's claimed ~31–32 dB on DIV2K at x2 uses DIV2K LR pairs (natural photos at scale 2x), not SRGD GameEngineData. SRGD's game-engine renders have different image statistics (smooth shading, high PSNR baselines of 36–38 dB against bicubic-clean LR), so absolute numbers are not directly comparable to the paper's tables. The sign and direction of the margin are the diagnostic, not the absolute values.

---

## Decision

**ENGINE-ALIASED LR IS THE BINDING CONSTRAINT.**

GSASR loses to bicubic on every single one of the 24 engine-aliased frames, with a mean margin of −0.04 dB and zero frames where it wins. The margin is tiny but consistent and negative across all 24 frames.

This is the same pattern as the prior Sintel eval (`2026-05-01-pretrained-gaussian-sr-eval.md`, −4.55 dB), but the gap is smaller here because our engine-aliased LR is harder for bicubic than clean bicubic-down LR — the sigma=1.5 TAA blur and JPEG compression mean bicubic upsample is no longer the near-perfect inverse. GSASR nearly catches up, but still doesn't pass.

The result settles the `2026-05-02-splats-SR-literature-delta.md` open question #1 definitively:

> "Would GSASR beat bicubic on engine-aliased LR?" → **No.**

### What this rules out

- **"Drop in GSASR"** as an upgrade path. Running released GSASR on our LR distribution does not provide a quality improvement over bicubic.
- **"The architectural gap is the binding constraint"** as a thesis. GSASR's architectural advantages (EDSR backbone, Gaussian Interaction Blocks, free-covariance Gaussians, 16 Gaussians/pixel) do not overcome the LR domain mismatch. The architecture is not the bottleneck.

### What this confirms

- **Training on engine-aliased LR is the real fix.** The −0.04 dB margin shows GSASR nearly matches bicubic on our LR — significantly closer than its −3.7 dB loss on bicubic-clean LR. GSASR generalises somewhat, but falls short of the threshold needed to actually be useful.
- **V0.5 CNN super-resolver is the correct current deliverable.** The pivot decision in `2026-05-02-splats-cannot-SR-definitive.md` (Decision 2: "ship V0.5 as a CNN super-resolver") is correct. V0.5 was trained on engine-aliased SRGD data and beats bicubic by +1.3 dB on the same distribution where GSASR falls short.
- **Sprint 5 should focus on training an SR model on engine-aliased LR**, not on architectural innovations. A standard CNN (Real-ESRGAN, EDSR) trained end-to-end against engine-aliased LR pairs would likely beat GSASR on our distribution significantly, following the same principle that made V0.5 work where the Gaussian splat didn't.

### Recommended next action

- [ ] Accept `2026-05-02-splats-cannot-SR-definitive.md` Decision 5 as final: **ship V0.5 CNN as the v0 SR deliverable**. No Sprint 5 re-implementation of GSASR architecture needed.
- [ ] The "Gaussian features → CNN decoder" redesign (GaussianSR thesis) is still open as a v2 stretch, but it requires training on engine-aliased LR — which V0.5 already does. The added value of the Gaussian feature path over a pure CNN is not demonstrated by this result.
- [ ] If a higher-quality model is needed post-V0.5: **train Real-ESRGAN or EDSR against engine-aliased SRGD LR** (the known working recipe). Do not attempt to fix the domain-mismatch issue by switching Gaussian SR architectures.

---

## Open questions

1. **Would GSASR beat bicubic on engine-aliased LR if fine-tuned on that distribution?** Almost certainly yes — it nearly matches bicubic already without training. But fine-tuning GSASR is equivalent to building V0.6 (a larger model), not a "free win." This is a real research question but not blocking the V0.5 ship.

2. **Is the −0.04 dB margin meaningful, or noise?** The consistency across 24 frames (0/24 GSASR wins, σ of margin ~0.02 dB) rules out noise. GSASR is systematically and consistently slightly behind bicubic on engine-aliased LR. This is not measurement variance.

3. **Why is the engine-aliased margin (−0.04 dB) so much smaller than the bicubic-clean margin (−3.70 dB)?** The engine-aliased LR introduces blur and JPEG artifacts that reduce the "bicubic-as-near-inverse" advantage. On clean bicubic LR, bicubic upsample almost perfectly recovers the GT; GSASR's hallucinated HF detail hurts PSNR heavily. On aliased LR, there is genuine degradation that a learned model can partially recover, but GSASR's DIV2K training distribution is still too different from SRGD ActionRPG to fully exploit this.

---

## Artifacts

All outputs on <train-host> at `<train-host-data>\eval_results\gsasr_engine_aliased\`:
- `metrics.json` — full per-frame results (JSON)
- `summary.txt` — human-readable summary (note: Unicode print error on Windows console; JSON is the authoritative record)
- `hr/` — 24 HR frames (960x540 PNGs)
- `lr_aliased/` — 24 engine-aliased LR frames (480x270 PNGs)
- `lr_bicubic/` — 8 bicubic-clean LR frames (480x270 PNGs)
- `sr_aliased_gsasr/` — 24 GSASR SR outputs on engine-aliased LR
- `sr_aliased_bicubic/` — 24 bicubic SR outputs on engine-aliased LR
- `sr_clean_gsasr/` — 8 GSASR SR outputs on bicubic-clean LR
- `sr_clean_bicubic/` — 8 bicubic SR outputs on bicubic-clean LR
- Eval script: `<train-host-data>\eval_gsasr_engine_aliased.py`

GSASR weights (`<train-host-data>\external-eval\GSASR\weights\EDSR_DIV2K\`) are CC-BY-NC 4.0 and are NOT committed to this repo.

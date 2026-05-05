# 2026-05-04 — v5 pixel-temporal: v4-warmstart vs from-scratch A/B

**Status:** Run A (warm-start) in flight, ETA ~05:00 CDT 2026-05-05. Chain watchdog (remote PID 28596) auto-launches Run B (from-scratch) when A exits. Race-compare pending Codex's C16 script + manual closeout call.

## Question

Is the v4 backbone warm-start helping or hurting v5 pixel-temporal training on TartanAir?

## Why we're asking

v4 was trained on **SRGD** — 17 stylized UE5 game scenes. v5 is training on **TartanAir** — photorealistic outdoor sim (forests, mountains, ruins). The content distributions are noticeably different:

- SRGD: indoor / urban / stylized PBR
- TartanAir: outdoor / natural / photorealistic + real depth + real flow

Standard ML transfer-learning wisdom says warm-start almost always wins because low-level conv features (edges, color statistics, interpolation) generalize across content. But "almost always" isn't "always," and Cash's mid-flight observation that v5-temporal at step 4K looked **softer than bicubic on TartanAir foliage** suggests v4's SRGD prior might be net-negative on this content.

We can't know without running both.

## Hypothesis

Two competing hypotheses, neither dominant a priori:

**H1 (transfer wins):** v4 warm-start saves ~20-30K equivalent training steps. The SRGD bias is real but small and gets overwritten by phase 2's joint backbone fine-tune. Final warm-start model beats from-scratch by 0.5-1 dB PSNR / -5-15% LPIPS at step 80K.

**H2 (cargo-cult init):** v4 warm-start's SRGD bias is large enough that phase 1 is wasted (frozen backbone can't unlearn SRGD; head-only training plateaus early) and phase 2 spends most of its budget undoing v4 instead of learning TartanAir. From-scratch matches or beats warm-start by step 80K.

## Setup

### Run A (warm-start, current)

```bash
python scripts/sr_train_temporal.py \
  --output-dir   <train-host-data>/checkpoints/srcnn-v5-pixel-temporal \
  --warm-start   <train-host-data>/checkpoints/srcnn-prod-v4-lpips/step-00385000.pt \
  --tartanair-root <train-host-data>/datasets/tartanair_extracted \
  --held-out-envs oldtown \
  --max-steps    80000 \
  --device       cuda \
  --num-workers  4
```

Spawned 22:54 CDT 2026-05-04 (PID 24416 on `<train-host>`). Includes the channel-zero distshift fix from `b2fa647`.

### Run B (from-scratch, queued)

Identical except `--warm-start` removed. Spawned automatically by `<train-host-data>\checkpoints\v5-chain-watchdog.ps1` (PID 28596) when Run A's PID exits. Lands at `<train-host-data>/checkpoints/srcnn-v5-pixel-temporal-fromscratch`.

## What to compare

Same fixed held-out batch (manifest at `<train-host-data>/checkpoints/v5_held_out_manifest.json`, 64 pairs from TartanAir env `oldtown` only).

| Metric | Why it matters |
|---|---|
| **PSNR** | Pixel-fidelity reference; what optimizer is directly minimizing |
| **LPIPS** | Perceptual quality; correlates with viewer preference |
| **Temporal consistency** (warp `t -> t+1`, |delta|) | What v5 is supposed to deliver vs single-frame v4 |
| **Latency** | Both should be identical (same architecture); sanity check |

## Decision rule

The winner ships as v5. If they're statistically tied (paired Wilcoxon p > 0.05 on 3+ of 4 metrics), warm-start ships by default — the SRGD prior at minimum doesn't hurt and the run-time was no different.

## Cost

10 hours of wall time on the shared 3080 Ti (one extra run, sequential per Cash's "sequential GPU" directive). Cheap relative to the project-long value of knowing whether to warm-start v6+ from a previous v on a new dataset.

## Followups

1. **Codex C16:** the race-compare script (`scripts/sr_v5_race_compare.py`) was supposed to land for the v5-pixel-vs-Gaussian race — same structure works here for warm-start-vs-from-scratch. Pending Codex.
2. **Sprint 5 closeout memo:** will reference this A/B's verdict. If from-scratch wins, that's a significant finding for the v5 design memo.
3. **v6 / Sprint 6 implication:** if warm-start loses on a content shift this small (game -> sim), the pico distillation plan needs to consider whether to distill from a v4-style or fresh-trained teacher.

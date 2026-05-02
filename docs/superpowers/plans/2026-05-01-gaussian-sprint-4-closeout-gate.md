# Sprint 4 Close-Out Gate — Iso-Latency vs FSR / DLSS

**Plan update:** docs/superpowers/research-synthesis-2026-05-01.md, plan update 2.
**Position in pipeline:** runs at the end of Sprint 4 (param network trained); blocks the start of Sprint 5 (persistent canvas).

## Why this gate exists

The original graduation criterion compares OSS-Gaussian only against
OSSPico — a friendly baseline. External research review surfaced the
sharper question: *does the Gaussian param network produce
upscaling-only results that beat FSR 2 / DLSS Quality at iso-latency?*

If the answer is **yes**, the temporal canvas (Sprint 5) and frame
extrapolation (Sprint 6) are differentiators that compound the win over
DLSS 4. If the answer is **no**, the temporal canvas can't save us;
we'd be stacking advanced features on top of a network that's already
losing the spatial-quality fight. In that case we'd revisit Sprint 4
network architecture (per-tile attention head? hybrid CNN+transformer?
larger model?) before investing in Sprints 5–7.

Catching that branch *before* a 3-week temporal-canvas implementation +
1-week frame-gen sprint matters.

## What runs at the gate

1. **Train Sprint 4 network on the production dataset** (Sintel +
   TartanAir + HyperSim + SRGD weighted mix; defer Cyberpunk RenderDoc
   captures to post-Sprint 2). Targets all four tiers (pico → ultra)
   on Lambda H100. Cost: ~$60 critical-path per the Sprint 4 plan.
2. **Export each tier** to TensorRT INT8 on the 3080 Ti (Sprint 4 T4.10
   in the existing plan).
3. **Run the iso-latency benchmark harness** at
   `oss/gaussian/bench/closeout_gate.py` (this doc's deliverable):
   - Sintel test split (~100 frames held out from training).
   - Cyberpunk 2077 RenderDoc captures (if available; else skip
     game-specific evaluation).
   - For each (tier, target_resolution) cell, measure:
     - PSNR / SSIM / LPIPS-VGG / FLIP
     - Total wall-clock latency end-to-end on the 3080 Ti
     - VRAM peak
   - Comparators (from `oss/gaussian/bench/baselines.py`):
     - bicubic
     - lanczos
     - FSR 2 Quality
     - DLSS Quality
     - OSSPico (existing baseline, friendly comparator)
4. **Generate close-out report** at
   `docs/superpowers/sprint-4-closeout-report.md` populated by the
   harness with metric tables, per-scene breakdowns, and a verdict
   block.

## Acceptance criteria — gate decision tree

The verdict block follows this rubric:

```
At iso-latency (within ±15% of FSR 2 Quality wall-clock latency on RTX 3080 Ti):

  PSNR(Gaussian) > PSNR(FSR 2 Quality) by ≥ 0.5 dB           — STRONG GO
  PSNR(Gaussian) within 0.5 dB of FSR 2 Quality              — GO
  PSNR(Gaussian) within 1.0 dB of FSR 2 Quality              — CONDITIONAL GO
                                                                 (requires architectural revisit before
                                                                  Sprint 7 ports)
  PSNR(Gaussian) > 1.0 dB worse than FSR 2 Quality           — NO-GO
                                                                 (revisit network architecture first)

Subjective check on Cyberpunk frames:
  Visible artifacting in 0/N frames                          — pass
  Visible artifacting in 1-2/N frames                        — pass with notes
  Visible artifacting in 3+/N frames                         — fail (regardless of metrics)

Decision = AND(quantitative gate, subjective gate)
```

`STRONG GO` and `GO` proceed to Sprint 5. `CONDITIONAL GO` allows
Sprint 5 work to start but flags Sprint 7 (cross-platform ports) as
risky — we may not want to ship to Steam Deck if the upscaler is
already at parity with FSR. `NO-GO` halts forward progress; loop back
to Sprint 4 architecture variants until a `GO` is achievable.

## Why DLSS Quality is informational, not gating

DLSS uses NVIDIA Tensor Cores and is fundamentally faster on RTX
hardware than vendor-agnostic ML inference of the same network. We
report PSNR vs DLSS for context but do not gate on iso-latency vs
DLSS — we'd lose by construction on the latency axis. The relevant
question for OSS-Gaussian is "do we beat FSR 2 quality at FSR 2
latency?" — that's the cross-vendor comparison.

The DLSS comparison serves as a quality ceiling reference: if Gaussian
exceeds DLSS Quality on PSNR even with the latency penalty, the case
for shipping it as a vendor-agnostic alternative is overwhelming
regardless of the latency gap.

## Harness shape (pseudo-code)

```python
# oss/gaussian/bench/closeout_gate.py

from oss.gaussian.bench.baselines import REGISTRY as BASELINES
from oss.gaussian.bench.gaussian import gaussian_upscale  # T4.x exposes this

DATASETS = {"sintel_test": ..., "cyberpunk": ...}
TIERS = ["pico", "lite", "standard", "ultra"]
SCALES = [2.0, 3.0, 4.0]

def run_gate():
    rows = []
    for tier in TIERS:
        net = load_trained_network(tier)
        for ds_name, ds in DATASETS.items():
            for scale in SCALES:
                for baseline_name, baseline_cls in BASELINES.items():
                    metrics = score_baseline(baseline_cls(), ds, scale)
                    rows.append({"tier": tier, "ds": ds_name,
                                 "scale": scale, "method": baseline_name,
                                 **metrics})
                metrics = score_gaussian(net, ds, scale)
                rows.append({"tier": tier, "ds": ds_name,
                             "scale": scale, "method": "gaussian",
                             **metrics})
    write_report(rows, verdict_decision_tree(rows))
```

## Deliverables tracked at the gate

- [ ] Sprint 4 network trained for 4 tiers on Lambda H100
- [ ] TensorRT INT8 exports for each tier on 3080 Ti
- [ ] `oss/gaussian/bench/closeout_gate.py` implemented and runs against
      the test datasets
- [ ] `docs/superpowers/sprint-4-closeout-report.md` generated with full
      metric tables + verdict
- [ ] Code review pipeline (Sprint 1 T1.8 pattern) run over the Sprint 4
      diff
- [ ] User-visible verdict committed to repo before Sprint 5 kickoff

## What this gate is NOT

- Not the production graduation criterion (those are still in the
  design spec § 5).
- Not a hardware comparison (different GPUs render different speeds; we
  pin to 3080 Ti for this gate).
- Not a DLSS-replacement claim. NVIDIA hardware dominance on Tensor
  Cores is acknowledged.

## What happens if we're between GO and NO-GO

`CONDITIONAL GO` is the most likely outcome on first attempt because:
- Synthetic-data-only training has a domain gap vs game footage
- Sprint 4 is v0 of a novel architecture without the iteration tail of
  DLSS

In that case: take 1 week to architect a revised Sprint 4 (per-tile
attention head as documented in research-synthesis section 5 plan
update 3), retrain, re-run the gate. Then proceed to Sprint 5 with a
network that meets the bar.

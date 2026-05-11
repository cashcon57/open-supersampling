# H007 — HAT-Tiny Forward Time at 1080p LR (Real-World Upscale Case)

**Status:** `validated` — measured on idle RTX 3080 Ti; H006's contaminated 107.5 ms revised to a clean **54.5 ms** at 270×480 LR; new measurement of **1,890 ms** at 1920×1080 LR locks in that HAT-Tiny is unshippable for the dominant 1080p→1440p / 1080p→4K user case.
**Class:** Hardware measurement gate for v6.2 inference-vs-teacher decision.
**Filed:** 2026-05-11
**Source:** Reframing question — H006 measured at 270×480 LR (4× upscale to 1080p HR); the real-world dominant case is 1080p LR → 1440p HR (1.33×) or 1080p LR → 4K HR (2×). The backbone runs at LR feature resolution, so the LR shape is what dictates HAT cost, not the HR output.

## Claim

H006 measured HAT-Tiny FP16 eager at 107.548 ms median on a co-host-training-contaminated 3080 Ti, at 270×480 LR (the 540p-tier LR for 1080p HR output). Two things weakened that as a final number:

1. The GPU was not idle — the v6.1-pico-001 trainer was using the same card.
2. The LR shape did not match the realistic shipping target. Most end-user upscaling is 1080p LR → 1440p HR (Quality) or 1080p LR → 4K HR (Performance), not 540p LR → 1080p HR.

H007 redoes the measurement on a **clean idle GPU** and adds the **1920×1080 LR** input shape to settle the real-world cost.

## Result

Measured on `3080ti-windows` / RTX 3080 Ti, idle (trainer + held-out eval + viz daemon all paused, no other compute apps on the GPU per `nvidia-smi --query-compute-apps`), via `tests/perf/hat_tiny_bench.py`:

| Input shape | Use case | FP32 eager median ms | FP16 eager median ms |
|---|---|---:|---:|
| 1×9×270×480 | 540p LR → 1080p HR (H006 repro) | 123.610 | **54.539** |
| 1×9×1080×1920 | **1080p LR → 1440p / 4K HR (dominant real-world case)** | 21,419.862 | **1,890.575** |

Compile path (`torch.compile`) still fails on this Windows conda env (`image-gs`) because Triton is unavailable; the bench script's compile variant prints "Skipping" and emits eager-only numbers. The H006 caveat there still holds.

## Cross-checks

- **H006 reproduction.** H006 reported 107.548 ms median FP16 eager at 270×480 LR with co-host contamination. The idle re-run measures 54.539 ms — exactly half. Co-host training was eating ~50% of the available GPU time on H006's measurement. The H006 *verdict* (HAT too slow for inference at the 1.5 ms gate) is unaffected: 54.5 ms misses the 1.5 ms gate by 36×, the 3.0 ms nano-student gate by 18×.
- **Pixel-count scaling.** 1920×1080 / 480×270 = 16× more LR pixels. 1890.575 / 54.539 = 34.7× more time. HAT-Tiny scales roughly **2.2× worse than pure pixel ratio** at this size jump, not perfectly linear in HW as a fixed-window self-attention naïvely would predict. Two suspected contributors: (a) FP16 tensor-core utilization drops at larger feature maps because dispatch/memory overhead amortizes less well, and (b) the convolutional stems and merging operations grow with HR features. We do not need a precise model — what matters is that the real number is worse than linear extrapolation, not better.
- **Compared to ship budgets.** DLSS 4 FP8 path: ~1.5–2 ms at 1080p→4K. FSR 4 ML: ~1.5–2 ms. XeSS XMX: ~2–3 ms. HAT-Tiny FP16 eager at 1080p LR: **1,890 ms** — **~945× over the 2 ms budget**.
- **Optimistic native-stack speedup.** Best-case (4–8×) speedup from native CUDA + TensorRT FP8 + flash attention compiled with Triton: 236–473 ms. Still **118–236×** over budget.

## Verdict

HAT-Tiny is dead for end-user inference at the dominant 1080p-LR upscale case. The kill-HAT decision from H006 is reaffirmed at the real-world LR shape with a fresh, idle measurement.

HAT-Tiny remains the **research / teacher** backbone for v6.x training, where forward time per training step is not a shipping constraint. The end-user inference model must be a distilled student per the [Phase 4 council 1–4 ms budget reassessment](../2026-05-08-phase4-msframe-council/03-1-4ms-budget-reassessment.md): ≤1M-param student in TensorRT FP8 with custom cross-vendor kernels, trained against the v6.2-pico-002 (and later v6.x Standard / Heavy) teacher checkpoints.

This is not a future "we should consider distilling" — it's a present, hard latency wall. Every model currently on the public dashboard is a teacher.

## Procedure

Idle-GPU procedure used for H007 (sequence the next person should re-run unchanged):

1. SSH to `3080ti-windows`.
2. Identify all `python.exe` / supervisor processes with `wmic process get name,processid,commandline /format:csv | findstr -iE "python|supervisor"`.
3. Pause the held-out-eval supervisor first (so it does not respawn its eval child): `taskkill /PID <heldout-eval-supervisor.ps1 pid> /F`.
4. Kill the held-out eval process (the cuda one) and any inflight flicker / probe scripts.
5. Kill the trainer process tree: `taskkill /PID <trainer pid> /T /F` (workers die with the parent).
6. Verify `nvidia-smi` shows `--query-gpu=memory.used` near OS-only residual (~500–800 MiB on this machine) and `--query-compute-apps` lists no `python.exe`.
7. Run `tests/perf/hat_tiny_bench.py` at the desired shape via `bash -lc` over ssh.
8. Re-launch the trainer via `scripts\3080ti\launch-run.ps1` (auto-resumes from the latest checkpoint; this round lost 20 steps between step-52000 ckpt and the resume).
9. Re-launch the held-out-eval supervisor via `Invoke-CimMethod Win32_Process Create` (orphan-spawn).

## Evidence

- Harness: `tests/perf/hat_tiny_bench.py`
- Sibling memo: `docs/research/hypotheses/H006-hat-tiny-actual-ms.md`
- Decision reaffirmed: `docs/research/2026-05-08-phase4-msframe-council/03-1-4ms-budget-reassessment.md`

Reproduce (idle GPU required for clean numbers — see Procedure above):

```bash
python tests/perf/hat_tiny_bench.py --iters 50 --warmup 10 --height 270 --width 480 --no-compile
python tests/perf/hat_tiny_bench.py --iters 30 --warmup 5  --height 1080 --width 1920 --no-compile
```

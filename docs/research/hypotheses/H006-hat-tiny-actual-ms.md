# H006 — HAT-Tiny Actual Forward Time

**Status:** `validated` — measured on reachable 3080 Ti; optimistic <=1.5 ms inference claim rejected for the current PyTorch v6 HAT-Tiny implementation
**Class:** Hardware measurement gate for v6.2-pico student size
**Filed:** 2026-05-08
**Source:** Phase 4 council dispute over HAT-Tiny inference cost

## Claim

HAT-Tiny may be cheap enough to keep in the v6.2-pico inference graph if its actual FP16 forward time on reference NVIDIA hardware is <= 1.5 ms at 1080p-output-equivalent LR input.

Input contract:

- 9 channels: RGB + depth + motion + normals
- LR shape: `1x9x270x480`
- Model: actual `oss.sr.v6.hat.hat_tiny(in_channels=9)` backbone used by `--backbone hat-tiny` pico training
- Timing: `torch.cuda.Event`, 100 measured iterations after 10 warm-up iterations

Decision gate:

- <= 1.5 ms: keep HAT-Tiny in inference at 30 Hz amortized
- 1.5-3.0 ms: distill to about 1M student
- > 3.0 ms: use a <=0.4M nano student

## Result

Measured on `3080ti-windows` / RTX 3080 Ti in a disposable clone at `C:\Users\cashc\oss-hat-tiny-bench-20260508`:

| Variant | Median ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| FP32 eager | 408.730 | 408.730 | 438.086 | 442.418 |
| FP16 eager | 107.548 | 107.548 | 122.762 | 129.716 |

The default compile path was checked separately and failed because the Windows conda environment did not have a working Triton install. A direct run from `E:\oss-gaussian` measured FP16 median 109.089 ms with the same missing-Triton compile failure. A later eager-only run while the trainer remained resident measured FP16 median 191.609 ms, confirming the host was not clean enough for final latency publication.

Local CUDA/4070-mobile result: not available. The local machine exposed to this task was an Apple M3 Max MacBook Pro with no CUDA GPU. `venv-py312` has PyTorch 2.11.0 with MPS available but CUDA unavailable; system/.venv/venv Python environments did not have torch installed.

## Validation Notes

The 3080 Ti run is contaminated by an active `srcnn-v6.1-pico-001` training process on the same GPU. Immediately after the first run, `nvidia-smi` reported 96% GPU utilization and 9,272 MiB used. PyTorch also warned that the installed build lacks flash attention, and `torch.compile` could not run because Triton was unavailable.

Those caveats weaken the exact absolute latency number, but not the decision gate: the best observed FP16 median was 107.548 ms, far beyond the >3 ms cutoff.

## Verdict

The <=1.5 ms keep-HAT hypothesis is rejected for the current implementation.

Proceed with the council-side branch: remove HAT-Tiny from the inference hot path and target a <=0.4M nano student unless a later clean, compiled, idle-host run disproves this by more than an order of magnitude.

## Evidence

- Harness: `tests/perf/hat_tiny_bench.py`
- Memo: `docs/superpowers/experiments/2026-05-08-hat-tiny-forward-ms-measurement.md`

Reproduce:

```bash
python tests/perf/hat_tiny_bench.py --iters 100 --warmup 10
```

# CUDA Phase 4a Baseline

**Date:** 2026-05-08
**Status:** benchmark harness validated; production-shape baseline and Nsight
Compute report are blocked.

## What Ran

New harness:

```bash
python tests/cuda/bench_end_to_end_v6.py --shape smoke --warmup 3 --iterations 10
```

This validates the full v6.1 stage breakdown and JSON schema with a synthetic
16,384-Gaussian canvas. It is not the requested 1080p -> 4K production
baseline; production shape is blocked below.

## 16k Smoke Results

| Host | GPU | Backend | Total median / p90 | HAT median | Fusion median | Spawner median | Raster+composite median |
|---|---|---|---:|---:|---:|---:|---:|
| G14 | RTX 4070 Laptop, sm_89 | `oss_cuda` | 43.322 / 46.472 ms | 4.656 ms | 4.867 ms | 0.367 ms | 33.123 ms |
| 3080 Ti | RTX 3080 Ti, sm_86 | `gsplat` | 125.917 / 139.138 ms | 56.823 ms | 18.351 ms | 5.848 ms | 39.033 ms |

Cross-host delta for this smoke run: 3080 Ti measured 2.91x slower than G14.
This is not a hardware conclusion: the 3080 Ti run was concurrent with the
protected trainer PID 22712, uses older Torch without flash attention, and used
`gsplat` rather than the native `oss_cuda` path.

JSON artifact: `docs/coordination/bench-baseline-v6-e2e.json`.

## Production-Shape Blocker

Requested production shape:

- LR: 1920 x 1080 x 9
- HR: 3840 x 2160 x 3
- Canvas: 16,384 Gaussians
- Iterations: 10 warmup + 100 timed

Result on G14: one production-shape probe OOMed before completing the first
timed iteration. The failure occurs in `rasterizer_composite_head` after the
full-HR `canvas_hr`, `refined_hr`, and composite-head activations consume most
of the 8 GB VRAM.

Result on 3080 Ti: production probe was not run because the protected v6.1
trainer PID 22712 was active and using about 7.6 GB of the 12 GB card. Running
the 4K probe beside it would likely OOM or materially perturb training.

## Nsight Compute Status

No `tests/cuda/profile-v6-e2e.ncu-rep` exists yet.

Tooling probes:

- G14: `ncu` not found in PATH; no `ncu` binary found under `/opt`, `/usr`, or `/home/cashc`.
- 3080 Ti: `where.exe ncu` and `where.exe ncu-ui` found no binary; no `ncu.exe`
  found under `C:\Program Files\NVIDIA Corporation`.

Target capture command once Nsight Compute is installed and production shape
can run:

```bash
ncu --set full \
  --target-processes all \
  --launch-skip 10 \
  --launch-count 1 \
  --export tests/cuda/profile-v6-e2e.ncu-rep \
  python tests/cuda/bench_end_to_end_v6.py --shape prod
```

Raw export and parser:

```bash
ncu --import tests/cuda/profile-v6-e2e.ncu-rep --page raw --csv \
  > tests/cuda/profile-v6-e2e.raw.csv
python scripts/parse_ncu_profile.py tests/cuda/profile-v6-e2e.raw.csv
```

## Kernel Table

Kernel-level values are unavailable until `ncu` is installed and the prod
forward fits/runs.

| Rank | Kernel | Time | Occupancy | Registers/thread | Bandwidth | Compute | Tensor core utilization |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Hot Kernel

Single hottest kernel: TODO pending NCU. From component wallclock only, the
hottest component is `rasterizer_composite_head`; the expected kernel-level hot
spot remains the rasterizer accumulation kernel (`rasterize_sum` /
`rasterize_forward`) once NCU is available.

Tensor core utilization: TODO pending NCU. Expected current utilization for
the rasterizer path is 0%; HAT/composite convolution kernels may use tensor
cores depending on PyTorch/cuDNN choices and dtype.

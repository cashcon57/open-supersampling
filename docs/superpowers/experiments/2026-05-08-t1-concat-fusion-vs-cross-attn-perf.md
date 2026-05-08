# 2026-05-08 T1 - ConcatFusion vs PixelGaussianFusion Forward Perf

**Status:** PARTIAL CUDA RUN on 3080 Ti; ConcatFusion measured, PixelGaussianFusion OOMed.

## Question

Compare `oss/sr/v6/concat_fusion.py::ConcatFusion` with
`oss/sr/v6/cross_attention.py::PixelGaussianFusion` at the same effective pixel
resolution.

Acceptance gate:

- PASS: `ConcatFusion` forward median `<=` `PixelGaussianFusion` forward median
- FAIL: `ConcatFusion` forward median `>` `PixelGaussianFusion` forward median
- NOT RUN: no CUDA-event timings available

If a CUDA run produces FAIL, a GitHub issue is required for the acceptance
failure. No issue was created here because the acceptance gate was not evaluated.

## Harness

Benchmark file:

```text
scripts/bench_concat_fusion_vs_cross_attn.py
```

Default synthetic case:

- Batch: `B=2`
- Pixel feature shape for both modules: `2x180x540x960`
- `ConcatFusion` fields at HR:
  - `F`: `2x180x540x960`
  - `G`: `2x4x540x960`
  - `m`: `2x1x540x960`
  - `I_base`: `2x3x540x960`
  - `depth`: `2x1x540x960`
  - `MV`: `2x2x540x960`
- `PixelGaussianFusion` tokens:
  - Gaussian tokens: `2x1024x64`
  - `num_heads=6`
  - `window_size=16`

The script runs eager forward passes under `torch.inference_mode()`. It uses 10
warmup iterations and 100 measured iterations by default. Measurements use
`torch.cuda.Event(enable_timing=True)`. The primary comparison is median
milliseconds per forward for matching dtype variants.

The script attempts fp32 and fp16 by default. TF32 is disabled by default; it can
be enabled with `--allow-tf32`.

## Implementation Comparison

`ConcatFusion` concatenates the full-resolution pixel feature tensor with the
HR canvas/readout fields, then applies `1x1 conv -> depthwise 3x3 conv ->
SqrSwish -> 1x1 conv` and adds the result back to `F`.

`PixelGaussianFusion` partitions the same pixel feature resolution into
non-overlapping `16x16` windows. For `540x960`, the implementation pads height
to `544`, yielding `34 * 60 = 2040` windows per image, or `4080` windows for
`B=2`. Each window query attends to the same `K=1024` Gaussian tokens after
token projection, then applies an output projection and MLP residual.

## Local Environment

```text
ProductName: macOS
ProductVersion: 26.3.1
BuildVersion: 25D771280a
Architecture: arm64
```

Available interpreter with PyTorch:

```text
./venv-py312/bin/python
python 3.12.13
torch 2.11.0
cuda_available False
torch_cuda_runtime None
mps_available True
```

System/repo default Python and two other venvs did not have PyTorch installed.

## Commands Run

```bash
python3 scripts/bench_concat_fusion_vs_cross_attn.py
```

Output:

```text
NOT RUN: PyTorch is unavailable: No module named 'torch'
```

```bash
./.venv/bin/python scripts/bench_concat_fusion_vs_cross_attn.py
```

Output:

```text
NOT RUN: PyTorch is unavailable: No module named 'torch'
```

```bash
./venv/bin/python scripts/bench_concat_fusion_vs_cross_attn.py
```

Output:

```text
NOT RUN: PyTorch is unavailable: No module named 'torch'
```

```bash
./venv-py312/bin/python scripts/bench_concat_fusion_vs_cross_attn.py
```

Output:

```text
NOT RUN: CUDA is unavailable; this benchmark requires CUDA events.
```

Syntax check:

```bash
python3 -m py_compile scripts/bench_concat_fusion_vs_cross_attn.py
```

Output: no output, exit code 0.

## Results

Measured on `Cash-PC` / RTX 3080 Ti, torch 2.4.1, CUDA runtime 12.4:

| Dtype | ConcatFusion median ms | PixelGaussianFusion median ms | Verdict |
|---|---:|---:|---|
| fp32 | 17.513 | OOM | NOT COMPARABLE |
| fp16 | 9.622 | OOM | NOT COMPARABLE |

ConcatFusion details:

| Dtype | Median ms | Mean ms | p95 ms | p99 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| fp32 | 17.513 | 17.375 | 18.170 | 19.582 | 2179.9 |
| fp16 | 9.622 | 11.412 | 14.283 | 29.177 | 1779.6 |

PixelGaussianFusion failed before producing timing:

- fp32: attempted allocation of about 23.91 GiB.
- fp16: attempted allocation of about 11.95 GiB on a 12 GiB card, after other
  allocations/reservations left no free VRAM.

## Verdict

**NOT COMPARABLE** vs the strict acceptance gate
`ConcatFusion forward <= PixelGaussianFusion forward at same effective resolution`.

The intended baseline cannot run at the requested same-resolution synthetic
case on the 12 GB 3080 Ti. This still supports the architectural concern that
global window-to-all-Gaussians cross-attention is not viable in the hot path,
but it does not produce the requested median/p99 comparison for
PixelGaussianFusion.

To evaluate on a CUDA host, run:

```bash
python scripts/bench_concat_fusion_vs_cross_attn.py --warmup 10 --iters 100
```

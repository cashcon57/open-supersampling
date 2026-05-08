# 2026-05-08 - H002 Rasterizer Latent-Rank Performance

**Owner:** B2

**Verdict:** FAIL

## Question

Does the existing v6 rasterizer's `latent_rank` path make R=4 forward rendering at
least 4x faster than the R=64 baseline on the synthetic benchmark?

Acceptance gate:

- PASS: R=4 forward median latency is >=4.0x faster than R=64 forward median.
- FAIL: R=4 forward median latency is <4.0x faster than R=64 forward median.
- NOT RUN: CUDA/PyTorch benchmark prerequisites are unavailable.

## Harness

Benchmark script:

```text
scripts/sr_v6_rasterizer_rank_microbench.py
```

The harness imports the existing implementation:

```python
from oss.sr.v6.rasterizer import V6Rasterizer
```

Synthetic default case:

- N=4096 Gaussians
- H=540, W=960
- token_dim=64
- latent ranks R in `{4, 8, 16, 32, 64}`
- random `xy`, positive-definite conic generated from random scale/rotation,
  and random `feat`
- 10 warmup iterations, then 100 timed iterations
- CUDA events for forward timings
- CUDA events for backward timings when backward is supported
- median and nearest-rank p99 milliseconds
- speedup vs R=64 median

The v6 rasterizer API consumes `scale`/`rot`, not raw conic coefficients. The
script still constructs the random conic implied by the synthetic scale/rotation
geometry so the generated case matches the requested `xy/conic/feat` setup while
using the production `V6Rasterizer(latent_rank=R)` contract.

## 3080 Ti Environment

Measured on the reachable 3080 Ti host:

- Host: `Cash-PC`
- GPU: NVIDIA GeForce RTX 3080 Ti
- Driver: 595.79
- PyTorch: 2.4.1
- CUDA runtime: 12.4
- Renderer backend: `gsplat`

Command:

```bash
python scripts/sr_v6_rasterizer_rank_microbench.py --device cuda:0 --warmup 10 --iters 100
```

## Local Environment

Local checkout:

```text
/Users/cashconway/OpenSuperSampling
```

CUDA/NVIDIA status:

```text
nvidia-smi not found
```

System Python:

```text
python torch unavailable: ModuleNotFoundError("No module named 'torch'")
```

`venv-py312`:

```text
venv-py312 torch 2.11.0 cuda False mps True
```

## Commands Run

Syntax check:

```bash
python -m py_compile scripts/sr_v6_rasterizer_rank_microbench.py
```

Result: passed with no output.

System Python benchmark attempt:

```bash
python scripts/sr_v6_rasterizer_rank_microbench.py
```

Output:

```text
NOT RUN: torch unavailable: ModuleNotFoundError: No module named 'torch'
```

`venv-py312` benchmark attempt:

```bash
./venv-py312/bin/python scripts/sr_v6_rasterizer_rank_microbench.py
```

Output:

```text
NOT RUN: torch.cuda.is_available() is False
```

## Results

CUDA timing on the 3080 Ti:

| Direction | R | Median ms | p99 ms | Speedup vs R=64 | Status |
|---|---:|---:|---:|---:|---|
| Forward | 4 | 3.802 | 4.420 | 3.18x | OK |
| Forward | 8 | 3.970 | 4.700 | 3.05x | OK |
| Forward | 16 | 5.566 | 6.437 | 2.17x | OK |
| Forward | 32 | 7.449 | 8.790 | 1.63x | OK |
| Forward | 64 | 12.107 | 13.464 | 1.00x | OK |
| Backward | 4 | 6.230 | 7.108 | 3.62x | OK |
| Backward | 8 | 6.772 | 8.055 | 3.33x | OK |
| Backward | 16 | 9.537 | 11.221 | 2.36x | OK |
| Backward | 32 | 13.466 | 15.111 | 1.67x | OK |
| Backward | 64 | 22.553 | 24.952 | 1.00x | OK |

## Gate

**FAIL:** R=4 forward speedup was `3.18x` versus the required `>=4.0x`.

Blocker filed:

- https://github.com/cashcon57/open-supersampling/issues/15

## Repro

From a CUDA-capable checkout:

```bash
python scripts/sr_v6_rasterizer_rank_microbench.py
```

Optional flags:

```bash
python scripts/sr_v6_rasterizer_rank_microbench.py --iters 100 --warmup 10
python scripts/sr_v6_rasterizer_rank_microbench.py --no-backward
```

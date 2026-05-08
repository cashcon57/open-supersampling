# 2026-05-08 - H005 Foreground Second-Pass Rasterizer Cost

**Owner:** B5

## Question

Can v6 render a foreground/mid-interval second rasterizer pass cheaply enough for
OSS-FX frame extrapolation?

Gate:

- **PASS:** two-pass total median time <= 1.5x the single display-frame pass
- **FAIL:** two-pass total median time > 1.5x the single display-frame pass
- **NOT RUN:** no real CUDA timing was produced

## Harness

Benchmark file:

```bash
scripts/bench_v6_fg_second_pass_cost.py
```

The harness imports the production v6 wrapper:

```python
from oss.sr.v6.rasterizer import V6Rasterizer
```

Synthetic workload:

- N = 4096 live Gaussians
- H = 540, W = 960
- R in {4, 8, 64}
- one synthetic `CanvasState`, all entries active
- pass 1 uses alpha = 1.0 display-frame motion offset
- pass 2 uses alpha = 0.5 mid-interval extrapolated motion offset
- timings use `torch.cuda.Event`
- reported gate metric is `total_median_ms / pass1_median_ms`

Recommended CUDA command:

```bash
python scripts/bench_v6_fg_second_pass_cost.py --iters 100 --warmup 10
```

## 3080 Ti Result

**Verdict: FAIL.**

Measured on `Cash-PC` / RTX 3080 Ti, torch 2.4.1, CUDA runtime 12.4, using the
renderer's default `gsplat` backend.

Command:

```bash
python scripts/bench_v6_fg_second_pass_cost.py --warmup 10 --iters 100
```

| R | Pass 1 median ms | Pass 2 median ms | Total median ms | Total / pass 1 | Verdict |
|---:|---:|---:|---:|---:|---|
| 4 | 5.654 | 5.789 | 11.832 | 2.09x | FAIL |
| 8 | 6.236 | 5.925 | 12.352 | 1.98x | FAIL |
| 64 | 12.467 | 12.590 | 25.042 | 2.01x | FAIL |

Local non-CUDA attempts:

```bash
python3 -m py_compile scripts/bench_v6_fg_second_pass_cost.py
```

Result: passed.

```bash
python3 scripts/bench_v6_fg_second_pass_cost.py --iters 1 --warmup 0
```

Output:

```json
{
  "gate": "two-pass total median <= 1.5x pass1 median",
  "ranks": [4, 8, 64],
  "reason": "torch unavailable: ModuleNotFoundError: No module named 'torch'",
  "shape": {"H": 540, "N": 4096, "W": 960},
  "status": "NOT RUN"
}
```

```bash
./venv-py312/bin/python scripts/bench_v6_fg_second_pass_cost.py --iters 1 --warmup 0
```

Output:

```json
{
  "gate": "two-pass total median <= 1.5x pass1 median",
  "ranks": [4, 8, 64],
  "reason": "torch.cuda.is_available() is false",
  "shape": {"H": 540, "N": 4096, "W": 960},
  "status": "NOT RUN"
}
```

```bash
./.venv/bin/python scripts/bench_v6_fg_second_pass_cost.py --iters 1 --warmup 0
./venv/bin/python scripts/bench_v6_fg_second_pass_cost.py --iters 1 --warmup 0
```

Both reported:

```json
{
  "gate": "two-pass total median <= 1.5x pass1 median",
  "ranks": [4, 8, 64],
  "reason": "torch unavailable: ModuleNotFoundError: No module named 'torch'",
  "shape": {"H": 540, "N": 4096, "W": 960},
  "status": "NOT RUN"
}
```

## Acceptance

**Verdict vs gate: FAIL.**

The second pass is effectively another full rasterizer pass; the measured
two-pass total is about 2x pass 1, not <=1.5x.

Blocker filed:

- https://github.com/cashcon57/open-supersampling/issues/15

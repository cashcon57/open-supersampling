# 2026-05-08 - T5 Student vs HAT-Tiny Forward-Time Check

**Verdict: PASS.** The 3080 Ti CUDA run measured the student at more than 2x faster than the HAT-Tiny FP16 eager baseline.

## Question

Can the T5 student backbone meet the acceptance gate of being at least 2x faster than the measured HAT-Tiny FP16 eager baseline?

Student under test:

```python
StudentBackbone(in_channels=9, channels=48, depth=4, out_features=180)
```

Synthetic input:

```text
B=2, C=9, H_LR=540, W_LR=960, dtype=fp16, device=CUDA
```

Harness:

```text
scripts/sr_bench_t5_student_backbone.py
```

Warmup and measurement:

```text
warmup=10, measured iterations=100
```

## HAT-Tiny Baseline

Source: `docs/superpowers/experiments/2026-05-08-hat-tiny-forward-ms-measurement.md`

Primary comparison number from the documented 3080 Ti disposable-clone run:

| Model | Variant | Median ms | p99 ms |
|---|---|---:|---:|
| HAT-Tiny | FP16 eager | 107.548 | 129.716 |

The 2x-faster student gate is therefore:

```text
student median <= 107.548 / 2 = 53.774 ms
```

## Local Run Attempts

Default system Python:

```bash
python3 scripts/sr_bench_t5_student_backbone.py
```

Output:

```text
NOT RUN: torch unavailable (ModuleNotFoundError: No module named 'torch')
```

Repo Python with torch:

```bash
./venv-py312/bin/python scripts/sr_bench_t5_student_backbone.py
```

Output:

```text
NOT RUN: CUDA unavailable
torch: 2.11.0
python: 3.12.13
host: Cashs-MacBook-Pro.local
```

## Result

| Model | Median ms | p99 ms | Gate | Verdict |
|---|---:|---:|---:|---|
| T5 StudentBackbone | 30.806 | 32.822 | <= 53.774 ms median | PASS |

3080 Ti command:

```bash
python scripts/sr_bench_t5_student_backbone.py --warmup 10 --iters 100
```

Output summary:

```text
gpu: NVIDIA GeForce RTX 3080 Ti
median_ms: 30.806
p99_ms: 32.822
hat_tiny_fp16_median_ms: 107.548
student_2x_gate_median_ms: 53.774
verdict: PASS
```

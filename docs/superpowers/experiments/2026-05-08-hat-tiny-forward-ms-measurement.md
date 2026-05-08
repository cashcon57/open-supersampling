# 2026-05-08 — HAT-Tiny Forward-Time Measurement

**Status:** measured on reachable 3080 Ti under active training load; 4070 mobile not available from this checkout.

## Question

Phase 4 council argued that v6.1 HAT-Tiny must be removed from inference because the backbone would cost 2-3 ms. The skeptical counter-claim was that the FP16 Tensor-Core-eligible path might be closer to 0.5-1.0 ms.

Gate:

- <= 1.5 ms: keep HAT-Tiny in inference at 30 Hz amortized
- 1.5-3.0 ms: distill to about 1M student
- > 3.0 ms: council direction stands; target a <= 0.4M nano student

## Harness

Benchmark file: `tests/perf/hat_tiny_bench.py`

The harness imports the actual v6 HAT-Tiny factory used by pico training:

```python
from oss.sr.v6.hat import hat_tiny
```

It runs `hat_tiny(in_channels=9)` on a synthetic 9-channel LR input of shape `1x9x270x480`, matching RGB + depth + motion + normals at 1080p-output-equivalent LR resolution. Timings use `torch.cuda.Event` over 100 measured iterations after 10 warm-up iterations.

## Hardware Tested

### 3080ti-windows

- Host: `3080ti-windows` / `Cash-PC`
- OS: Windows 10 `10.0.26220`
- GPU: NVIDIA GeForce RTX 3080 Ti
- VRAM: 12,884,377,600 bytes reported by PyTorch; `nvidia-smi` reports 12,288 MiB
- SM: 8.6, 80 SMs
- Driver: 595.79
- PyTorch: 2.4.1
- CUDA runtime: 12.4
- `nvidia-smi` CUDA version: 13.2
- cuDNN: 90100
- Python: 3.11.10 in `image-gs` for the disposable-clone run
- TF32 matmul: disabled (`torch.backends.cuda.matmul.allow_tf32 = False`)

Important caveat: this was not an idle-host measurement. The active trainer

```text
C:\Users\cashc\Miniconda3\envs\image-gs\python.exe scripts\sr_train_v6.py
  --output-dir E:\checkpoints\srcnn-v6.1-pico-001
  --tartanair-root E:\datasets\tartanair_extracted
  --backbone hat-tiny ...
```

remained resident on the GPU during the benchmark. Immediately after the first run, `nvidia-smi` reported 9,272 MiB used and 96% GPU utilization. A later sample showed 7,904 MiB used and 0% instantaneous utilization, with the trainer still resident. Treat the absolute timings as contaminated high-side measurements, not clean idle latency.

### Local Machine

- Host: local macOS checkout
- Machine: MacBook Pro `Mac15,9`
- CPU/SoC: Apple M3 Max, 16 CPU cores
- GPU: Apple M3 Max, 40 GPU cores, Metal 4
- RAM: 64 GB
- CUDA/NVIDIA: unavailable; `nvidia-smi` and `nvcc` unavailable
- PyTorch:
  - system `python3`: torch unavailable
  - `./.venv/bin/python`: torch unavailable
  - `./venv/bin/python`: torch unavailable
  - `./venv-py312/bin/python`: torch 2.11.0, MPS available, CUDA unavailable

No local CUDA timing was produced. The requested 4070 mobile CUDA target was not attached to this environment.

## Results

### 3080 Ti Disposable-Clone Run

The remote subagent created a disposable clone at:

```text
C:\Users\cashc\oss-hat-tiny-bench-20260508
```

and copied only `tests/perf/hat_tiny_bench.py` into it. The clone was at commit `45bcbe83ee116e4e0c5e294d1b5d001aa6a052b7`.

Command:

```powershell
Set-Location C:\Users\cashc\oss-hat-tiny-bench-20260508
conda run -n image-gs python tests/perf/hat_tiny_bench.py --iters 100 --warmup 10 --no-compile
```

| Variant | Median ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| FP32 eager | 408.730 | 408.730 | 438.086 | 442.418 |
| FP16 eager | 107.548 | 107.548 | 122.762 | 129.716 |

This is the best observed FP16 median and the primary number for the decision gate. PyTorch warned that it was not compiled with flash attention.

### 3080 Ti Run 1

Command:

```powershell
cd E:\oss-gaussian
C:\Users\cashc\Miniconda3\envs\image-gs\python.exe tests\perf\hat_tiny_bench.py --iters 100 --warmup 10
```

Notes:

- PyTorch warned that it was not compiled with flash attention.
- `torch.compile` failed for both FP32 and FP16 because the environment did not have a working Triton install.
- FP16 eager is the primary measured number.

| Variant | Median ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| FP32 eager | 224.615 | 224.615 | 262.370 | 267.619 |
| FP32 compile | failed: missing Triton | | | |
| FP16 eager | 109.089 | 109.089 | 123.214 | 131.562 |
| FP16 compile | failed: missing Triton | | | |

### 3080 Ti Run 2, No Compile

Command:

```powershell
cd E:\oss-gaussian
C:\Users\cashc\Miniconda3\envs\image-gs\python.exe tests\perf\hat_tiny_bench.py --iters 100 --warmup 10 --no-compile
```

| Variant | Median ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| FP32 eager | 282.269 | 282.269 | 446.935 | 2026.460 |
| FP16 eager | 191.609 | 191.609 | 204.945 | 211.551 |

This second run was more heavily disturbed by the resident training process and should not replace Run 1 as the primary figure.

## Comparison to Council Claim

The council's 2-3 ms claim was not reproduced. The measured FP16 eager HAT-Tiny backbone on the reachable 3080 Ti was far slower: best observed median 107.548 ms, with the caveat that the GPU was not idle.

Even allowing for training-load contamination and missing flash/Triton compile support, this result does not support the optimistic <= 1.5 ms hypothesis for the current PyTorch v6 HAT-Tiny implementation. The current implementation is made of many window-attention and convolution operations at LR image resolution; it is not behaving like one ideal dense Tensor Core GEMM.

## Verdict

**Verdict: nano <0.4M student for the current v6.2-pico inference path.**

Do not keep the current HAT-Tiny backbone in inference. The available 3080 Ti measurement is not clean enough to publish as final hardware latency, and the 4070 mobile target still needs a true CUDA run, but the measured numbers are so far above the 3 ms gate that the student-model decision should not wait on the optimistic 0.5-1.0 ms estimate.

## Reproducibility

From a CUDA-capable checkout:

```bash
python tests/perf/hat_tiny_bench.py --iters 100 --warmup 10
```

For eager-only timing when `torch.compile`/Triton is unavailable:

```bash
python tests/perf/hat_tiny_bench.py --iters 100 --warmup 10 --no-compile
```

Recommended follow-up for clean publication numbers:

1. Stop or move the active `srcnn-v6.1-pico-001` trainer before measuring.
2. Install a PyTorch/Triton stack where `torch.compile` works.
3. Run the same harness on the actual 4070 mobile CUDA machine.

# 2026-05-08 - H004 L2 Hit-Rate Measurement

**Owner:** D2
**Verdict:** NOT RUN

## Question

Can the existing rasterizer keep the synthetic Gaussian workload cache-friendly
enough to support the H004 L2-resident scheduler claim?

Requested isolated profile:

- N=4096 Gaussians
- H=540, W=960
- R=64 current rasterizer, forward only
- Nsight Compute metrics:
  - `l2_request_hit_rate.pct`
  - `l2_request_miss_rate.pct`
  - `dram_bytes_read.sum`
  - `gpu_time_active.avg`
- Repeat at R=8 if feasible

## Local Environment

Local checkout:

```text
/Users/cashconway/OpenSuperSampling
```

Host:

```text
Darwin Cashs-MacBook-Pro.local 25.3.0 Darwin Kernel Version 25.3.0: Wed Jan 28 20:54:55 PST 2026; root:xnu-12377.91.3~2/RELEASE_ARM64_T6031 arm64
macOS 26.3.1 (a), build 25D771280a
Apple M3 Max
```

Python and CUDA status:

```text
venv/bin/python: torch unavailable: ModuleNotFoundError: No module named 'torch'
venv-py312/bin/python: torch 2.11.0, torch.cuda.is_available() False, torch.version.cuda None, MPS available True
```

NVIDIA tooling status:

```text
ncu: command not found
nv-nsight-cu-cli: command not found
nvidia-smi: command not found
nvcc: command not found
```

## Profiling Harness

Added a dedicated forward-only Nsight target:

```text
scripts/sr_h004_l2_profile_probe.py
```

It uses the requested synthetic dimensions by default:

- N=4096
- H=540, W=960
- token_dim=64
- latent rank R=8 or R=64
- forward path through `oss.sr.v6.rasterizer.V6Rasterizer`

The production rasterizer selects the native CUDA extension only when tensors
are on CUDA and `OSS_USE_CUDA_KERNELS` enables the rasterizer path. The relevant
build/run hooks are:

```bash
pip install --no-build-isolation -e ./oss/cuda --force-reinstall
OSS_USE_CUDA_KERNELS=rasterizer python scripts/sr_h004_l2_profile_probe.py --rank 64
```

## Commands Attempted

Tool discovery:

```bash
command -v ncu
command -v nv-nsight-cu-cli
command -v nvidia-smi
command -v nvcc
```

Output:

```text
no paths returned
```

Direct version probes:

```bash
ncu --version
nv-nsight-cu-cli --version
nvidia-smi
nvcc --version
```

Output:

```text
zsh:1: command not found: ncu
zsh:1: command not found: nv-nsight-cu-cli
zsh:1: command not found: nvidia-smi
zsh:1: command not found: nvcc
```

System Python benchmark attempt:

```bash
python scripts/sr_v6_rasterizer_rank_microbench.py --n 4096 --h 540 --w 960 --token-dim 64 --ranks 64,8 --warmup 10 --iters 100 --no-backward
```

Output:

```text
NOT RUN: torch unavailable: ModuleNotFoundError: No module named 'torch'
```

Dedicated H004 probe attempt:

```bash
python scripts/sr_h004_l2_profile_probe.py --rank 64
```

Output:

```text
NOT RUN: torch unavailable: ModuleNotFoundError: No module named 'torch'
```

PyTorch venv benchmark attempt:

```bash
venv-py312/bin/python scripts/sr_v6_rasterizer_rank_microbench.py --n 4096 --h 540 --w 960 --token-dim 64 --ranks 64,8 --warmup 10 --iters 100 --no-backward
```

Output:

```text
NOT RUN: torch.cuda.is_available() is False
```

## Results

No Nsight Compute values were produced on this machine.

| Scenario | l2_request_hit_rate.pct | l2_request_miss_rate.pct | dram_bytes_read.sum | gpu_time_active.avg | Status |
|---|---:|---:|---:|---:|---|
| R=64, forward only | n/a | n/a | n/a | n/a | NOT RUN |
| R=8, forward only | n/a | n/a | n/a | n/a | NOT RUN |

## Blocker

This host is macOS on Apple Silicon. It has no NVIDIA GPU, no CUDA runtime, no
CUDA-enabled PyTorch build, and no Nsight Compute CLI in PATH. Nsight Compute
cannot profile the CUDA rasterizer here.

## Queued CUDA Host Run

Run this from a Linux or Windows NVIDIA CUDA host with Nsight Compute installed
and visible as `ncu`.

First, build the CUDA environment:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -e .
python -m pip install --no-build-isolation -e ./oss/cuda --force-reinstall
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Then profile R=64:

```bash
OSS_USE_CUDA_KERNELS=rasterizer ncu \
  --target-processes all \
  --metrics l2_request_hit_rate.pct,l2_request_miss_rate.pct,dram_bytes_read.sum,gpu_time_active.avg \
  --csv \
  --log-file /tmp/oss-h004-r64-ncu.csv \
  python scripts/sr_h004_l2_profile_probe.py \
    --n 4096 --h 540 --w 960 --token-dim 64 --rank 64 \
    --warmup 10 --iters 100
```

Then profile R=8:

```bash
OSS_USE_CUDA_KERNELS=rasterizer ncu \
  --target-processes all \
  --metrics l2_request_hit_rate.pct,l2_request_miss_rate.pct,dram_bytes_read.sum,gpu_time_active.avg \
  --csv \
  --log-file /tmp/oss-h004-r8-ncu.csv \
  python scripts/sr_h004_l2_profile_probe.py \
    --n 4096 --h 540 --w 960 --token-dim 64 --rank 8 \
    --warmup 10 --iters 100
```

If profiling overhead is too high, reduce to `--warmup 2 --iters 5` for the
Nsight run and keep the unprofiled 100-iteration microbench separately for
latency. Do not claim a PASS/FAIL for H004 from the reduced run unless the
same kernel names and metric availability are confirmed in the CSV.

## Gate

**NOT RUN:** H004's L2 acceptance gate cannot be evaluated locally. No measured
claim is made for L2 hit rate, DRAM bytes read, or active GPU time.

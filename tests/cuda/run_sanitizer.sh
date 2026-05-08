#!/usr/bin/env bash
set -e

PY=${OSS_PYTHON:-$HOME/miniforge3/envs/oss-cuda/bin/python}
SAN=${OSS_SANITIZER:-/opt/cuda/bin/compute-sanitizer}
SCRIPT=tests/cuda/sanitizer_smoke.py
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

for tool in memcheck racecheck initcheck synccheck; do
  echo "=== compute-sanitizer --tool=$tool ==="
  "$SAN" --tool="$tool" --error-exitcode=42 \
    --print-limit 50 --leak-check full "$PY" "$SCRIPT"
done

echo "all sanitizer tools exited 0"

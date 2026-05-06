#!/usr/bin/env bash
#
# cloud_bootstrap.sh — fresh-instance setup for a Lambda Labs / RunPod
# / Vast cloud GPU node. Run this once per new cloud instance, then
# launch training with cloud_train_v6.sh.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/cashcon57/open-supersampling/main/scripts/cloud_bootstrap.sh | bash
#
# Or after cloning manually:
#   git clone https://github.com/cashcon57/open-supersampling.git
#   cd open-supersampling
#   ./scripts/cloud_bootstrap.sh
#
# Steps:
#   1. Clone repo if not already in it
#   2. Create venv-py312 + install requirements
#   3. Download TartanAir + Hypersim (the big one; ~400 GB total)
#      via huggingface-cli or the project's existing download scripts
#   4. Verify GPU count + CUDA + bf16 support
#   5. Print the cloud_train_v6.sh launch command
#
# Assumes Ubuntu 22.04 / 24.04 base image with Python 3.12 + CUDA 12+
# (Lambda Labs default; RunPod PyTorch 2.x template).

set -euo pipefail

REPO_URL="https://github.com/cashcon57/open-supersampling.git"
REPO_DIR="${REPO_DIR:-$HOME/open-supersampling}"
DATASETS_ROOT="${DATASETS_ROOT:-/workspace/datasets}"

echo "[$(date -u +%H:%M:%S)] cloud_bootstrap starting"
echo "  REPO_DIR        $REPO_DIR"
echo "  DATASETS_ROOT   $DATASETS_ROOT"

# --- 1. clone or update repo ---
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout main
git pull --ff-only

# --- 2. venv + deps ---
if [[ ! -x "venv-py312/bin/python" ]]; then
  python3.12 -m venv venv-py312
  ./venv-py312/bin/pip install --upgrade pip wheel
fi
./venv-py312/bin/pip install -e .
./venv-py312/bin/pip install lpips pytorch-wavelets

# --- 3. datasets ---
mkdir -p "$DATASETS_ROOT"
cd "$DATASETS_ROOT"
if [[ ! -d "tartanair_extracted" ]]; then
  echo "Downloading TartanAir (~200 GB) via the official azcopy..."
  echo "  See https://theairlab.org/tartanair-dataset/ for credentials."
  echo "  ALTERNATIVELY mount your existing TartanAir copy as ./tartanair_extracted"
  echo "  Skipping auto-download — set up TartanAir manually then re-run."
fi
if [[ ! -d "ml-hypersim" ]]; then
  echo "Hypersim (~200 GB) — apple/ml-hypersim repo's download_data.py is the official path."
  echo "  Skipping auto-download — set up Hypersim manually then re-run."
fi

# --- 4. GPU verification ---
cd "$REPO_DIR"
./venv-py312/bin/python - <<'EOF'
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} ({p.total_memory // 1024**3} GB)")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
EOF

# --- 5. launch hint ---
NUM_GPUS=$(./venv-py312/bin/python -c 'import torch; print(torch.cuda.device_count())')
echo
echo "[bootstrap done] To launch training:"
echo "  ./scripts/cloud_train_v6.sh ${NUM_GPUS} /workspace/checkpoints/srcnn-v6-heavy-cloud-001"
echo
echo "To resume from a previous checkpoint, add --resume:"
echo "  ./scripts/cloud_train_v6.sh ${NUM_GPUS} /workspace/checkpoints/srcnn-v6-heavy-cloud-001 --resume"

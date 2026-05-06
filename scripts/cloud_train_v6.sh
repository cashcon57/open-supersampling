#!/usr/bin/env bash
#
# cloud_train_v6.sh — launch v6 training on a cloud GPU node.
#
# Designed for Lambda Labs / RunPod / Vast 4× or 8× A100 80GB / H100 80GB
# single-node multi-GPU instances. Single-node only — no inter-node DDP
# (would need InfiniBand and rendezvous setup).
#
# Usage on the cloud node, after cloning the repo + creating venv:
#   ./scripts/cloud_train_v6.sh <NUM_GPUS> [output-dir] [--resume]
#
# Examples:
#   ./scripts/cloud_train_v6.sh 4
#   ./scripts/cloud_train_v6.sh 8 /workspace/checkpoints/srcnn-v6-heavy-cloud-001
#   ./scripts/cloud_train_v6.sh 8 /workspace/checkpoints/srcnn-v6-heavy-cloud-001 --resume
#
# Defaults assume:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - 80 GB VRAM per GPU (uses memo-canonical batch=4/rank × patch=256)
#   - bf16 (Ampere+ all support natively)
#   - Datasets at /workspace/datasets/{tartanair_extracted,ml-hypersim}
#
# Dataset paths can be overridden:
#   DATASETS_ROOT=/data/oss ./scripts/cloud_train_v6.sh 8

set -euo pipefail

NUM_GPUS="${1:-1}"
OUTPUT_DIR="${2:-/workspace/checkpoints/srcnn-v6-heavy-cloud-001}"
EXTRA_ARGS=()
shift 2 2>/dev/null || true
EXTRA_ARGS+=("$@")

DATASETS_ROOT="${DATASETS_ROOT:-/workspace/datasets}"
TARTANAIR_ROOT="${DATASETS_ROOT}/tartanair_extracted"
HYPERSIM_ROOT="${DATASETS_ROOT}/ml-hypersim"
LOG_FILE="${OUTPUT_DIR}/train.log"

if [[ ! -d "$TARTANAIR_ROOT" ]]; then
  echo "FATAL: TartanAir not found at ${TARTANAIR_ROOT}" >&2
  echo "  Set DATASETS_ROOT=<dir> or symlink the datasets in." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# Memo §6 effective batch target = 16. Each rank carries batch_per_rank,
# with grad_accum=1 typically (DDP synchronizes per micro-step). Keep
# per-rank batch in [1..4] for a 80 GB GPU at patch=256.
case "$NUM_GPUS" in
  1)  BATCH=4;  GRAD_ACCUM=4;  PATCH=256 ;;  # effective 16
  2)  BATCH=4;  GRAD_ACCUM=2;  PATCH=256 ;;  # effective 16
  4)  BATCH=4;  GRAD_ACCUM=1;  PATCH=256 ;;  # effective 16 (memo recipe)
  8)  BATCH=2;  GRAD_ACCUM=1;  PATCH=256 ;;  # effective 16; safer per-rank batch
  *)
    echo "Unsupported NUM_GPUS=${NUM_GPUS} (use 1, 2, 4, or 8)" >&2
    exit 2
    ;;
esac

# Linear LR scaling with effective batch ratio. Memo base_lr=2e-4 at
# effective batch=16. We keep effective batch fixed at 16 so LR stays
# at 2e-4. If you push effective batch higher (e.g. by raising BATCH
# above), scale LR linearly: new_lr = 2e-4 * (new_effective / 16).
BASE_LR=2e-4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL on single-node uses NVLink/PCIe automatically. Tweak if needed.
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Prefer the venv-py312 interpreter if present, else system python3.
PY="$(pwd)/venv-py312/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

cat <<EOF | tee "$LOG_FILE"
[cloud_train_v6] $(date -u +%Y-%m-%dT%H:%M:%SZ)
  NUM_GPUS         $NUM_GPUS
  OUTPUT_DIR       $OUTPUT_DIR
  TARTANAIR_ROOT   $TARTANAIR_ROOT
  HYPERSIM_ROOT    $HYPERSIM_ROOT
  PER-RANK BATCH   $BATCH
  GRAD_ACCUM       $GRAD_ACCUM
  PATCH            $PATCH
  EFFECTIVE BATCH  $((BATCH * GRAD_ACCUM * NUM_GPUS))
  BASE_LR          $BASE_LR
  EXTRA_ARGS       ${EXTRA_ARGS[*]:-(none)}
  PY               $PY
EOF

# torchrun for single-node multi-GPU. --standalone picks an ephemeral
# rendezvous endpoint on localhost.
"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  scripts/sr_train_v6.py \
    --output-dir "$OUTPUT_DIR" \
    --tartanair-root "$TARTANAIR_ROOT" \
    --hypersim-root "$HYPERSIM_ROOT" \
    --backbone hat-l \
    --max-steps 300000 \
    --warmup-steps 20000 \
    --T0 50000 \
    --num-restarts 5 \
    --first-ckpt-step 100 \
    --ckpt-every 5000 \
    --base-lr "$BASE_LR" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --patch-size "$PATCH" \
    --trajectory-length 4 \
    --num-workers 8 \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

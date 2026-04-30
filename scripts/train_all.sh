#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-<home>/open-reconstruction-suite/venv-py312/bin/python}"
DATA_DIR="${DATA_DIR:-data/bistro_mvp}"
OUT_BASE="${OUT_BASE:-results}"

# 1. Render dataset (skip if already present)
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "==> Rendering dataset to $DATA_DIR ..."
  "$PYTHON" scripts/render_dataset.py --scene bistro --views 4 --out "$DATA_DIR"
else
  echo "==> Dataset present at $DATA_DIR; skipping render."
fi

# 2. Train ORD standalone
echo "==> Training ORD ..."
"$PYTHON" -m ors.train.train_ord --data "$DATA_DIR" --out "$OUT_BASE/ord" --epochs 20

# 3. Train ORU standalone
echo "==> Training ORU ..."
"$PYTHON" -m ors.train.train_oru --data "$DATA_DIR" --out "$OUT_BASE/oru" --epochs 20

# 4. Train paired (warm-start ORD)
echo "==> Training paired (warm-start from ORD) ..."
"$PYTHON" -m ors.train.train_paired \
  --data "$DATA_DIR" --out "$OUT_BASE/paired" \
  --ord-ckpt "$OUT_BASE/ord/ord.pth" --epochs 10

echo "==> All checkpoints in $OUT_BASE/"

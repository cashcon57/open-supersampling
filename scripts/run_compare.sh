#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-<home>/open-reconstruction-suite/venv-py312/bin/python}"
DATA_DIR="${DATA_DIR:-data/bistro_mvp}"
OUT_BASE="${OUT_BASE:-results}"

"$PYTHON" -m ors.valuation.compare \
  --ord-ckpt "$OUT_BASE/ord/ord.pth" \
  --oru-ckpt "$OUT_BASE/oru/oru.pth" \
  --paired-ckpt "$OUT_BASE/paired/paired.pth" \
  --data "$DATA_DIR" \
  --out "$OUT_BASE/comparison.csv"

echo "==> Comparison CSV at $OUT_BASE/comparison.csv"
column -s, -t < "$OUT_BASE/comparison.csv" | head

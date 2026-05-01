#!/usr/bin/env bash
# Downloads MPI Sintel training set (clean + final pass + optical flow)
# Usage: ./scripts/download_sintel.sh /data/sintel
set -euo pipefail
DEST="${1:-./data/sintel}"
mkdir -p "$DEST"
# MPI Sintel official URLs
BASE="http://files.is.tue.mpg.de/sintel"
for split in training_clean training_final training_extras; do
    fname="${split}.zip"
    if [ ! -f "$DEST/$fname" ]; then
        echo "Downloading $fname..."
        curl -L "$BASE/$fname" -o "$DEST/$fname"
    fi
    echo "Extracting $fname..."
    unzip -q -n "$DEST/$fname" -d "$DEST"
done
echo "Done. Sintel at $DEST"

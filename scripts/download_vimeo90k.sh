#!/usr/bin/env bash
# Downloads Vimeo-90K septuplet dataset (~33GB)
# Usage: ./scripts/download_vimeo90k.sh /data/vimeo90k
set -euo pipefail
DEST="${1:-./data/vimeo90k}"
mkdir -p "$DEST"
URL="http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
FNAME="$DEST/vimeo_septuplet.zip"
if [ ! -f "$FNAME" ]; then
    echo "Downloading Vimeo-90K (~33GB)..."
    curl -L "$URL" -o "$FNAME"
fi
echo "Extracting..."
unzip -q -n "$FNAME" -d "$DEST"
echo "Done. Vimeo-90K at $DEST"

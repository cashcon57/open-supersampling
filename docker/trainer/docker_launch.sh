#!/usr/bin/env bash
#
# Trainer entrypoint inside the oss-trainer container.
#
# Pulls latest origin/main on every container start so commits land without
# rebuilding the image. The code lives in a bind-mounted directory (see
# docker-compose.yml or the docker run -v invocation in
# "Starting up after wipe.md"), so the git operations happen on host-owned
# files inside the container.
#
set -e
cd /workspace/oss-gaussian
git fetch origin main
git reset --hard origin/main
echo "[entrypoint] HEAD: $(git log -1 --format='%h %s')"
exec python -u -X faulthandler scripts/sr_train_v7.py \
  --tartanair-root /datasets/tartanair_extracted \
  --output-dir /checkpoints/srcnn-v7.0-pico-005 \
  --steps 100000 --batch-size 2 --device cuda --log-every 50 \
  --ckpt-every 500 --ckpt-warmup-steps 100,500,1000,2000 \
  --backbone-kind hat_tiny --curriculum --enable-parent-child \
  --parent-child-drift-rate 0.05 --canvas-capacity 16384 \
  --max-hr-crop 384 --no-compile

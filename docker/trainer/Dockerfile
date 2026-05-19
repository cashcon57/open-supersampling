FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Build deps for flash-attn (only needed if compiling from source) and git for
# the launcher's runtime pull-latest behavior. ninja-build is harmless to include
# and useful for any future torch.compile inductor codegen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python deps. Kept in one layer so the rebuild is a single pip resolve.
RUN pip install --no-cache-dir \
    packaging wheel setuptools \
    lpips einops numpy pillow tqdm tensorboardX matplotlib \
    onnxruntime

# Flash-attn 2: prefer a pre-built wheel in the build context to avoid a
# 5-15 minute source compile that spikes RAM hard enough to OOM-kill dockerd
# under WSL2. See "Starting up after wipe.md" for how to obtain the wheel
# (either Dao-AILab's official Linux release artifact at
# https://github.com/Dao-AILab/flash-attention/releases, or a fresh source
# build with MAX_JOBS=2 inside a separate container).
#
# Wheel filename pattern:
#     flash_attn-<VERSION>-cp<PYVER>-cp<PYVER>-linux_x86_64.whl
# For torch 2.4.1+cu124 + Python 3.11 the version that has been tested is
# flash_attn-2.7.0.post2.
COPY flash_attn-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/flash_attn-*.whl && rm /tmp/flash_attn-*.whl

WORKDIR /workspace/oss-gaussian
ENV PYTHONPATH=/workspace/oss-gaussian

# Bind-mounted repo is owned by the host UID; permit git to operate on it.
RUN git config --global --add safe.directory '*'

COPY --chmod=755 docker_launch.sh /usr/local/bin/oss-train

# Health: trainer is considered alive iff history.jsonl has been written to
# in the last 5 minutes. The 20-minute start-period covers dataset enum +
# checkpoint load + first batch warmup. 3 retries x 2-minute interval means
# a stuck-but-alive trainer flips to "unhealthy" after ~6 minutes of silence.
HEALTHCHECK --interval=2m --timeout=10s --start-period=20m --retries=3 \
  CMD test $(($(date +%s) - $(stat -c %Y /checkpoints/srcnn-v7.0-pico-005/history.jsonl 2>/dev/null || echo 0))) -lt 300

CMD ["oss-train"]

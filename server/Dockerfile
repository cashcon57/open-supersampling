# OSS Capture Ingest server image.
#
# Slim Python base, no torch, no compilers. Boto3 + FastAPI + uvicorn only.
# Built image is ~180 MB and cold-starts in <2 s on Fly.io shared-cpu-1x.
#
# Build (from repo root):
#   docker build -f server/Dockerfile -t oss-capture-ingest:dev .
#
# Run (env file with R2 creds):
#   docker run --rm -p 8080:8080 --env-file .secrets/r2-credentials.env \
#     oss-capture-ingest:dev

FROM python:3.12-slim AS base

# Non-root user — the server has no need for root, and Fly.io / most
# managed runtimes flag root containers in security audits.
RUN useradd --system --uid 1001 --no-create-home --shell /sbin/nologin oss

WORKDIR /app

# Install deps first for layer caching.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy only what the server actually imports — keeps the image lean and
# means a docs / training-script change doesn't bust the build cache.
COPY server/ /app/server/
COPY pyproject.toml README.md /app/

# Token store + dedup-LRU live in /data so they survive container
# restarts when bound to a Fly.io volume (see fly.toml).
ENV OSS_CAPTURE_TOKEN_STORE=/data/oss-capture-tokens.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /data && chown oss:oss /data

USER oss

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).read()" \
   || exit 1

CMD ["python", "-m", "server.oss_capture_ingest.main", "serve", "--host", "0.0.0.0", "--port", "8080"]

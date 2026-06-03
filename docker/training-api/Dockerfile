FROM docker:cli
RUN apk add --no-cache python3 curl
COPY scripts/training_api.py /app/training_api.py
WORKDIR /app
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1
ENTRYPOINT ["python3", "training_api.py"]

#!/usr/bin/env bash
# Smoke-train ORU-Pico with synthetic random sequences (no NoiseBase needed).
# Useful for verifying the full forward/backward/checkpoint cycle on a fresh
# checkout. Completes in <2 minutes on Apple Silicon CPU.
set -euo pipefail
PYTHON="${PYTHON:-<home>/open-reconstruction-suite/venv-py312/bin/python}"
"$PYTHON" -m ors.train.train_pico --smoke-test --out results/pico_smoke --sequence-length 4

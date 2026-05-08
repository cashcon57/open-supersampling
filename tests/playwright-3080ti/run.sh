#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
npx playwright test --project=3080ti-chromium "$@"

#!/usr/bin/env bash
set -euo pipefail

metrics=$1
shift
mkdir -p "$(dirname "$metrics")"
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits --loop-ms=1000 > "$metrics" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
"$@"

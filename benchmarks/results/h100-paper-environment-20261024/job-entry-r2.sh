#!/usr/bin/env bash
# Invoked only by the root coordinator's single finite allocated-H100 Job.
set -euo pipefail
R=/inspire/ssd/project/chemicalreaction/czxs25220150/h100-paper-environment-20261024
source /opt/generativeqc-reuse/activate.sh
export PATH="$R/env/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export GENERATIVEQC_JIT_SCCACHE_ROOT="$R/jit-slots"
export PIP_CACHE_DIR="$R/pip-cache" TMPDIR="$R/tmp"
exec "$R/env/bin/python" -B "$R/scripts/gate-wait-v2/gate_wait.py" \
  --adapter "$R/scripts/adapter-prepared-v4/qz_smoke.py" \
  --adapter-sha256 e57fcbdffb689ec637346105ad5a976fe57dd29df5ed26d35eb21c3c4a419496 \
  --prepared-plan "$R/assets/prepared-smoke-plan-r2.json" \
  --prepared-plan-sha256 0b0dd82bfe010e6fe0b2cedec4abfc06dd835e2c8a5a60e10542cc15e53f8657 \
  --output "$R/output/h100-smoke-gate-20261024-9dcc-r2" \
  --release-marker "$R/output/h100-smoke-gate-20261024-9dcc-r2/release.json" \
  --job-name h100-paper-smoke-20261024-9dcc-r2 --excluded-node qb-prod-gpu095 \
  --wait-seconds 300 --process-cap 3600 --job-cap 14400

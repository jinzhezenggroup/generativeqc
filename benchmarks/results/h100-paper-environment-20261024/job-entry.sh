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
exec "$R/env/bin/python" -B "$R/scripts/gate-wait-v1/gate_wait.py" \
  --adapter "$R/scripts/adapter-prepared-v3/qz_smoke.py" \
  --adapter-sha256 a540bc5a6da30f3b605a9a56c91668ac8a896b44126972c733e111420f46400b \
  --prepared-plan "$R/assets/prepared-smoke-plan.json" \
  --prepared-plan-sha256 51fc2d6c071aa4209aa639174f3198a0cd53308ab203b155b04cbfd8dc0fe682 \
  --output "$R/output/h100-smoke-gate-20261024-9dcc-r1" \
  --release-marker "$R/output/h100-smoke-gate-20261024-9dcc-r1/release.json" \
  --excluded-node qb-prod-gpu095 \
  --wait-seconds 300 --process-cap 3600 --job-cap 14400

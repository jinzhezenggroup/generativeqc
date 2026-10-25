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
exec "$R/env/bin/python" -B "$R/scripts/gate-wait-v3/gate_wait.py" \
  --adapter "$R/scripts/adapter-prepared-v5/qz_smoke.py" \
  --adapter-sha256 66bd120eea89abf44a5a4e4a77616afc91979c7fb67822eab1a58011e6bb7b66 \
  --prepared-plan "$R/assets/prepared-smoke-plan-r2-v5.json" \
  --prepared-plan-sha256 cae38b70227e56a66e7dcda455a83b0d95b832ebcc33270f4ed9786b45cc5941 \
  --output "$R/output/h100-smoke-gate-20261024-9dcc-r2" \
  --release-marker "$R/output/h100-smoke-gate-20261024-9dcc-r2/release.json" \
  --job-name h100-paper-smoke-20261024-9dcc-r2 --excluded-node qb-prod-gpu095 \
  --wait-seconds 300 --process-cap 3600 --job-cap 14400

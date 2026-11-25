#!/usr/bin/env bash
# Run only the enrolled second-round points on the production build.
set -euo pipefail
output=${1:?new output directory}
cards=${2:?allocated card count}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plan=${3:-"$root/plan.json"}
assets=$(dirname -- "$root")
env_root="$assets/h100-portability/20261027-fd06/env"
toolkit=/opt/generativeqc-reuse/cuda-12.9
provider="$env_root/lib/python3.11/site-packages/scipy_openblas32/lib"
test -s "$root/build-receipts/identity.json"
test "$(cat "$root/build-receipts/exit-code.txt")" = 0
export GENERATIVEQC_LIBRARY="$root/build/libgenerativeqc.so"
export GENERATIVEQC_PROFILE=off
export SCCACHE_DIR="$assets/h100-portability/20261027-fd06/cache/sccache"
export PATH="$env_root/bin:$toolkit/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$root/build:$provider:$toolkit/lib64:$toolkit/targets/x86_64-linux/lib"
export PYTHONPATH="$root/source:$root/source/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$env_root/bin/python" -u "$root/run_round2.py" campaign "$output" "$plan" "$cards"

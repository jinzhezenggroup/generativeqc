#!/usr/bin/env bash
# Reproduce the qz execution in the project image and the allocated GPU job.
set -euo pipefail
dataset=${1:?HTBH38 or NHTBH38}
output=${2:?new output directory}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
assets=$(dirname -- "$root")
env_root="$assets/h100-portability/20261027-fd06/env"
toolkit=/opt/generativeqc-reuse/cuda-12.9
provider="$env_root/lib/python3.11/site-packages/scipy_openblas32/lib"
export GENERATIVEQC_LIBRARY="$root/build/libgenerativeqc.so"
export GENERATIVEQC_PROFILE=off
export GENERATIVEQC_STATIONARY_CACHE="$root/runtime-cache-$dataset"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$root/jit-owners-$dataset"
export SCCACHE_DIR="$assets/h100-portability/20261027-fd06/cache/sccache"
export PATH="$env_root/bin:$toolkit/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$root/build:$provider:$toolkit/lib64:$toolkit/targets/x86_64-linux/lib"
export PYTHONPATH="$root/source:$root/source/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ $dataset = shard ]]; then
    exec "$env_root/bin/python" -u "$root/run_shards.py" "$output" "$root/assignments.json" 4
fi
exec "$env_root/bin/python" -u "$root/run_gpu.py" campaign "$dataset" "$output"

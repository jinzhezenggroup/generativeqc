#!/usr/bin/env bash
# Use the pinned build and a task-local JIT owner in the paper H100 image.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
assets=$(dirname -- "$root")
runtime="$assets/h100-portability/20261027-fd06"
toolkit=/opt/generativeqc-reuse/cuda-12.9
provider="$runtime/env/lib/python3.11/site-packages/scipy_openblas32/lib"
export PATH="$runtime/env/bin:$toolkit/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$root/build:$provider:$toolkit/lib64:$toolkit/targets/x86_64-linux/lib"
export PYTHONPATH="$root/source:$root/source/python"
export GENERATIVEQC_LIBRARY="$root/build/libgenerativeqc.so"
export GENERATIVEQC_PROFILE=off
job=${GQC_QZ_JOB_NAME:?unique job name for runtime cache ownership}
export GENERATIVEQC_STATIONARY_CACHE="$root/runtime-cache/$job"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$root/jit-owners/$job"
export SCCACHE_DIR="$runtime/cache/sccache"
export GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK=50000000000000
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$runtime/env/bin/python" -u "$root/run_gpu.py" campaign "$@"

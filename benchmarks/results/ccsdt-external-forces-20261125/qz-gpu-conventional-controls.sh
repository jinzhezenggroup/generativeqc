#!/usr/bin/env bash
# Acquire bounded conventional small-case repeats on the frozen master build.
set -euo pipefail
root=$PWD
out="$root/$1"
shift
if [[ "$#" = 0 ]]; then
    set -- --timeout 1200 --cases water ammonia formaldehyde C1 C2 C3 --repeats 2
fi
base=$(dirname "$(dirname "$root")")
env_root="$base/h100-portability/20261027-fd06"
binary_root="$root/run-gpu-master-v1/build"
cuda=/opt/generativeqc-reuse/cuda-12.9
provider="$env_root/env/lib/python3.11/site-packages/scipy_openblas32/lib"
export GENERATIVEQC_LIBRARY="$binary_root/libgenerativeqc.so"
export LD_LIBRARY_PATH="$binary_root:$provider:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export PATH="$env_root/env/bin:$cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$root/source/python:$root/source"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SCCACHE_DIR="$env_root/cache/sccache"
export GENERATIVEQC_STATIONARY_CACHE="$out/runtime-cache"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$out/jit-owners"
mkdir "$out"
"$env_root/env/bin/python" -B -u "${GQC_BENCHMARK_DRIVER:-run.py}" --source "$root/source" \
    --arm native-conv --output "$out/native-conv" --threads 1 "$@" > "$out/native-conv.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
arm=$1
label=$2
out=$3
shift 3
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
protocol="${GQC_PROTOCOL_DIR:-$root/protocol}"
scratch="${GQC_SCRATCH_DIR:-/tmp}"

case "$arm" in
    native-conv)
        export GENERATIVEQC_LIBRARY="$GQC_NATIVE_BUILD/libgenerativeqc.so"
        export PYTHONPATH="$GQC_NATIVE_SOURCE/python:$GQC_NATIVE_SOURCE"
        export LD_LIBRARY_PATH="$GQC_NATIVE_BUILD:$GQC_BLAS_LIB:$GQC_CUDA_ROOT/lib64:$GQC_CUDA_ROOT/targets/x86_64-linux/lib"
        export PATH="$GQC_CUDA_ROOT/bin:$(dirname "$GQC_PYTHON"):$PATH"
        export CUBLAS_WORKSPACE_CONFIG=:16:8
        export GENERATIVEQC_STATIONARY_CACHE="$out.runtime-cache"
        export GENERATIVEQC_JIT_SCCACHE_ROOT="$out.jit-owners"
        set -- --source "$GQC_NATIVE_SOURCE" --source-archive "$GQC_NATIVE_ARCHIVE" "$@"
        ;;
    psi4-conv|pyscf-conv) ;;
    *) exit 2 ;;
esac

exec "$GQC_PYTHON" -B -u "$root/run.py" \
    --protocol-dir "$protocol" --arm "$arm" --label "$label" \
    --output "$out" --scratch "$scratch" "$@"

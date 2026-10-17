#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a finite single-GPU srun allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm-assigned device visibility}"
: "${SOURCE_CHECKOUT:?set a checkout containing measured revision 51a87a04f}"
: "${NATIVE_ROOT:?set the measured linked native artifact directory}"
: "${REFERENCE_PYTHON:?set the recorded reference interpreter}"
native=$(cd -- "$NATIVE_ROOT" && pwd)
output=${OUTPUT:-.artifacts/benchmarks/wb97mv-hessian-response-reuse-$SLURM_JOB_ID.json}
[[ $output = /* ]] || output=$PWD/$output
printf '%s  %s\n' \
  86c398514c62eda62f0b21bf2e4b20d8f602058afb69894c3fac76e40aa7c76f "$native/generativeqc" \
  fd3ffd5a06e570029c3580590ee4e4c0a577b0bd9e66ad452055c982b50aa2ba "$native/libgenerativeqc.so.0.1.0" | sha256sum --check
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
git -C "$SOURCE_CHECKOUT" archive 51a87a04fdb8ef8e17bd8f6b848f2fcdc77f60af | tar -xf - -C "$work"
export PYTHONPATH="$work/python:$work/tests/python:$work"
export GENERATIVEQC_LIBRARY="$native/libgenerativeqc.so.0.1.0"
export LD_LIBRARY_PATH="$native:${REFERENCE_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DISABLE_PTX_JIT=1
cd "$work"
"$REFERENCE_PYTHON" benchmarks/compare_wb97mv_hessian_gpu4pyscf.py \
  --cases water-dimer formaldehyde water h2 --repeats 3 --timeout-seconds 180 \
  --gpu-memory-limit-mib 24576 --cli "$native/generativeqc" \
  --reference-python "$REFERENCE_PYTHON" \
  --source-commit 51a87a04fdb8ef8e17bd8f6b848f2fcdc77f60af --output "$output"

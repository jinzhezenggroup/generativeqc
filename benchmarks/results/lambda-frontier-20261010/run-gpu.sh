#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
campaign="$root/p2-lambda-frontier-master"
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$campaign/candidate:$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$campaign/candidate/source/python:$campaign/candidate/source:$deps"
export GENERATIVEQC_LIBRARY="$campaign/candidate/libgenerativeqc.so"
export GENERATIVEQC_DF_LAMBDA_CUDA_TEST=1
out="$campaign/qualification-$SLURM_JOB_ID"
if [[ "${1:-}" == memcheck ]]; then
  out="$campaign/memcheck-$SLURM_JOB_ID"
fi
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "HOSTNAME=$(hostname)" > "$out/allocation.txt"
cp "$0" "$out/reproduce.sh"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,memory.total,power.limit --format=csv > "$out/gpu.csv"
sha256sum "$GENERATIVEQC_LIBRARY" > "$out/binaries.sha256"
cd "$campaign/candidate/source"
ccache --version > "$out/ccache-version.txt"
ccache --show-stats > "$out/ccache-before.txt"
if [[ "${1:-}" == memcheck ]]; then
  compute-sanitizer --tool memcheck --target-processes all --error-exitcode=99 \
    --leak-check=full --launch-timeout=600 \
    /home/jzzeng/miniconda3/bin/python -m pytest \
    tests/python/test_df_lambda_core_reuse.py -k 'native_core_reuse or owner_local' \
    -q --tb=short --basetemp="$out/pytest" > "$out/memcheck.log" 2>&1
else
  /home/jzzeng/miniconda3/bin/python -m pytest \
  tests/python/test_df_lambda_core_reuse.py tests/python/test_df_lambda_matrix_audit.py \
  tests/python/test_df_lambda_provider_lifetime.py -q --tb=short --basetemp="$out/pytest" \
  > "$out/tests.log" 2>&1
fi
ccache --show-stats > "$out/ccache-after.txt"
printf '%s\n' complete > "$out/status.txt"

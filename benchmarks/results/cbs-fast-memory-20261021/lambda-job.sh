#!/usr/bin/env bash
set -euo pipefail
source /data/jzzeng/qc-cbs-fast-memory-20261021/candidate/environment.sh
export GENERATIVEQC_PROFILE=off GENERATIVEQC_DF_LAMBDA_CUDA_TEST=1
ccache --version > "$CBS_FAST_ROOT/results/lambda-cache-version.txt"
bash "$CBS_FAST_ROOT/python-preload.sh" -m pytest -q -rs tests/python/test_df_lambda_mixed_precision.py tests/python/test_df_lambda_mixed_scratch_composition.py tests/python/test_df_lambda_matrix_audit.py tests/python/test_df_lambda_serial_scratch.py > "$CBS_FAST_ROOT/results/pytest-lambda-gpu.log" 2>&1
ccache --show-stats > "$CBS_FAST_ROOT/results/lambda-cache-after.txt"

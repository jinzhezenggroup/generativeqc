#!/usr/bin/env bash
set -euo pipefail
source /data/jzzeng/qc-cbs-fast-memory-20261021/candidate/environment.sh
export GENERATIVEQC_PROFILE=off PYSCF_MAX_MEMORY=32768
ccache --version > "$CBS_FAST_ROOT/results/cache-version.txt"
ccache --show-stats > "$CBS_FAST_ROOT/results/cache-before.txt"
bash "$CBS_FAST_ROOT/python-preload.sh" /data/jzzeng/qc-cbs-fast-memory-20261021/water-qualification.py > "$CBS_FAST_ROOT/results/water-qualification.log" 2>&1
bash "$CBS_FAST_ROOT/python-preload.sh" -m pytest -q tests/python/test_full_df_cbs_native.py > "$CBS_FAST_ROOT/results/pytest-full-df.log" 2>&1
bash "$CBS_FAST_ROOT/python-preload.sh" -m pytest -q tests/python/test_df_lambda_mixed_precision.py tests/python/test_df_lambda_mixed_scratch_composition.py tests/python/test_df_lambda_matrix_audit.py tests/python/test_df_lambda_serial_scratch.py > "$CBS_FAST_ROOT/results/pytest-lambda.log" 2>&1
ccache --show-stats > "$CBS_FAST_ROOT/results/cache-after.txt"

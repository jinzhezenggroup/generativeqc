#!/usr/bin/env bash
set -euo pipefail
source /data/jzzeng/qc-cbs-fast-memory-20261021/candidate/environment.sh
export GENERATIVEQC_PROFILE=off PYSCF_MAX_MEMORY=32768
compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
  bash "$CBS_FAST_ROOT/python-preload.sh" -m pytest -q tests/python/test_full_df_cbs_native.py -k fast \
  > "$CBS_FAST_ROOT/results/memcheck.log" 2>&1

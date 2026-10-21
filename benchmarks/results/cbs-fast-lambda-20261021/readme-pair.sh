#!/usr/bin/env bash
set -euo pipefail
source /data/jzzeng/qc-cbs-fast-lambda-20261021/candidate/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_PROFILE=off
carbon="$1"
policies="standard fast"
if (( carbon % 2 == 0 )); then policies="fast standard"; fi
for policy in $policies; do
  output="$CBS_FAST_ROOT/readme-runs/$policy/$carbon"
  mkdir -p "$output"
  result=0
  timeout --kill-after=15 3600 "$CBS_FAST_ROOT/python-preload.sh" -m benchmarks.readme_cbs native \
    --qm7 /data/jzzeng/readme-cbs-20261013/qm7.mat --carbon "$carbon" \
    --policy "$policy" --hf-reference df --budget-mib 32768 --samples 1 \
    --source-revision 07c3933d7+lambda-fp32 \
    --output "$output/native.json" > "$output/native.log" 2>&1 || result=$?
  printf '{"exit_code":%s,"point_time_limit_seconds":3600}\n' "$result" > "$output/native.outcome"
done

#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-cbs-fast-lambda-20261021
for variant in baseline candidate; do
  source "$root/$variant/environment.sh"
  export GENERATIVEQC_PROFILE=off
  output="$CBS_FAST_ROOT/ablation/c6"
  mkdir -p "$output"
  "$CBS_FAST_ROOT/python-preload.sh" -m benchmarks.readme_cbs native \
    --qm7 /data/jzzeng/readme-cbs-20261013/qm7.mat --carbon 6 \
    --policy fast --hf-reference df --budget-mib 32768 --samples 1 \
    --source-revision "07c3933d7+$variant-lambda" \
    --output "$output/native.json" > "$output/native.log" 2>&1
done

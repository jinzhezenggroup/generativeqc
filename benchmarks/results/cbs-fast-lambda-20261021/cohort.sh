#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-cbs-fast-lambda-20261021
for carbon in 1 2 3 4 5; do
  bash "$root/candidate/readme-pair.sh" "$carbon"
done
for repeat in 1 2 3; do
  variants="baseline candidate"
  if (( repeat % 2 == 0 )); then variants="candidate baseline"; fi
  for variant in $variants; do
    source "$root/$variant/environment.sh"
    export GENERATIVEQC_PROFILE=off
    output="$CBS_FAST_ROOT/repeats/c4-$repeat"
    mkdir -p "$output"
    "$CBS_FAST_ROOT/python-preload.sh" -m benchmarks.readme_cbs native \
      --qm7 /data/jzzeng/readme-cbs-20261013/qm7.mat --carbon 4 \
      --policy fast --hf-reference df --budget-mib 32768 --samples 1 \
      --source-revision "07c3933d7+$variant-lambda" \
      --output "$output/native.json" > "$output/native.log" 2>&1
  done
done

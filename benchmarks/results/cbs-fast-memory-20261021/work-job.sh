#!/usr/bin/env bash
set -euo pipefail
cd /data/jzzeng/qc-cbs-fast-memory-20261021
for variant in baseline candidate; do
  output="/data/jzzeng/qc-cbs-fast-memory-20261021/$variant/results/c6-work"
  mkdir -p "$output"
  export GENERATIVEQC_DF_PROGRESS_TRACE="$output/trace.jsonl"
  test ! -e "$GENERATIVEQC_DF_PROGRESS_TRACE"
  bash run-one.sh "$variant" 6 fast c6-work
done

#!/usr/bin/env bash
set -euo pipefail
cd /data/jzzeng/qc-cbs-fast-memory-20261021
for repeat in 1 2 3; do
  variants='baseline candidate'
  if (( repeat == 2 )); then variants='candidate baseline'; fi
  for variant in $variants; do bash run-one.sh "$variant" 4 fast "c4-repeat-$repeat"; done
done
for variant in baseline candidate; do bash run-one.sh "$variant" 6 fast c6-ablation; done

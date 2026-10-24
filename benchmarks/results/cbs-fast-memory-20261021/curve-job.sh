#!/usr/bin/env bash
set -euo pipefail
cd /data/jzzeng/qc-cbs-fast-memory-20261021
for carbon in 1 2 3 4 5 6; do
  policies='standard fast'
  if (( carbon % 2 == 0 )); then policies='fast standard'; fi
  for policy in $policies; do bash run-one.sh candidate "$carbon" "$policy" "c$carbon-$policy"; done
done

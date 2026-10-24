#!/usr/bin/env bash
# C1 must be independently checked before expanding the reference cohort.
set -uo pipefail
root="${1:?absolute task root}"
threads="${2:?allocated numeric threads}"
memory_mb="${3:?PySCF memory limit in MB}"
output="${4:?fresh attempt output directory}"
cd "$root" || exit 1
failed=0
for carbon in 3 4 5 6 7; do
  bash "$root/run-oracle.sh" "$root" "$carbon" "$threads" "$memory_mb" "$output"
  code=$?
  printf 'C%s exit=%s\n' "$carbon" "$code"
  [[ "$code" == 0 ]] || failed=1
done
exit "$failed"

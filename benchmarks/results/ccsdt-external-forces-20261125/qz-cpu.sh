#!/usr/bin/env bash
# Called by srun from the task's shared storage directory.
set -euo pipefail
arm=$1
out=$2
mkdir "$out"
extra=()
case "$arm" in
    psi4-conv) env_root=cpu-env-v3 ;;
    psi4-df) env_root=cpu-env-v3; extra+=(--fd) ;;
    pyscf-conv) env_root=cpu-env-v4; extra+=(--fd) ;;
    *) exit 2 ;;
esac
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$env_root/bin/python" -B -u cpu-driver-v3/run.py --arm "$arm" --output "$out/records" \
    --threads 8 --memory-mib 24576 --timeout 2700 --expected-cpu 9462 \
    "${extra[@]}" > "$out/driver.log" 2>&1

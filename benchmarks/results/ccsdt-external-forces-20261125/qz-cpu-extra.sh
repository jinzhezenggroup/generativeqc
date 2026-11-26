#!/usr/bin/env bash
# Acquire the slowest points independently of the original sequential batch.
set -euo pipefail
arm=$1
out=$2
shift 2
case "$arm" in
    psi4-conv|psi4-df) env_root=cpu-env-v3 ;;
    pyscf-conv) env_root=cpu-env-v4 ;;
    *) exit 2 ;;
esac
mkdir "$out"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$env_root/bin/python" -B -u cpu-driver-v3/run.py --arm "$arm" --output "$out/records" \
    --expected-cpu 9462 "$@" > "$out/driver.log" 2>&1

#!/usr/bin/env bash
# Invoke via srun, from the campaign root, after confirming native inputs.
set -euo pipefail
root=$PWD
raw=$(realpath "$1")
run="$root/$2"
shift 2
mkdir "$run"
date -u +%FT%TZ > "$run/allocated-entrypoint.txt"
trap 'printf "%s\n" "$?" > "$run/exit-code.txt"; date -u +%FT%TZ > "$run/allocated-exit.txt"' EXIT
lscpu > "$run/cpu.txt"
grep -m1 '^model name' /proc/cpuinfo | grep -F 'CPU Max 9462' > "$run/validated-model.txt"
environment="$root/../issue-3166-20261125/cpu-env"
cp "$environment/packages.txt" "$run/packages.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/payload"
"$environment/venv/bin/python" -u "$root/${GQC_MINIMA_DRIVER:-check_minima.py}" "$root/payload" "$raw" "$run/raw" "$@"

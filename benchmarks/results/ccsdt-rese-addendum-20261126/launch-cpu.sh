#!/usr/bin/env bash
set -euo pipefail
root=$PWD
name=$1
shift
run="$root/cpu-$name"
mkdir "$run"
date -u +%FT%TZ > "$run/entry.txt"
trap 'printf "%s\n" "$?" > "$run/exit-code.txt"; date -u +%FT%TZ > "$run/exit.txt"' EXIT
lscpu > "$run/cpu.txt"
grep -m1 '^model name' /proc/cpuinfo | grep -F 'CPU Max 9462' > "$run/validated-model.txt"
environment="$root/../issue-3166-20261125/cpu-env"
cp "$environment/packages.txt" "$run/packages.txt"
cp -r "$root/cpu-payload" "$run/payload"
(cd "$run/payload" && sha256sum -c "$root/cpu-payload.sha256" && patch --batch -p1 -i "$root/parallel-minima.patch")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
args=()
phase=${GQC_CURVATURE_PHASE:-}
if test "$phase" = coarse || test -n "${GQC_CURVATURE_RESUME:-}"; then
  (cd "$run/payload" && patch --batch -p1 -i "$root/resume-minima.patch")
fi
if test -n "$phase"; then
  output="$root/cpu-$1/raw"
  PYTHONPATH="$run/payload" "$environment/venv/bin/python" -u "$root/split_curvature.py" \
    "$run/payload" "$root/native" "$output" "$1" "$phase" --workers "${GQC_CURVATURE_WORKERS:-5}"
  exit
fi
if test -n "${GQC_CURVATURE_RESUME:-}"; then
  # The previous Slurm step must be stopped before copying its point receipts.
  cp -r "$root/cpu-$GQC_CURVATURE_RESUME/raw" "$run/raw"
  args+=(--resume)
fi
for name in "$@"; do
  args+=(--molecule "$name")
done
"$environment/venv/bin/python" -u "$run/payload/check_df_minima.py" "$run/payload" "$root/native" "$run/raw" \
  --auxiliary "$run/payload/cc-pwcvtz-rifit.json" --workers "${GQC_CURVATURE_WORKERS:-5}" "${args[@]}"

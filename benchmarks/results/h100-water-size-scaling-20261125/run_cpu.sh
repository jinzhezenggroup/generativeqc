#!/usr/bin/env bash
# Run within an allocated srun step. Save every oracle exit even after a failure.
set -eu
root="$1"
python="$2"
output="$root/cpu"
mkdir "$output"
mkdir "$output/scratch"
export PYTHONPATH="$root/source:$root/source/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?Missing Slurm step}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYSCF_TMPDIR="$output/scratch"
cd "$root/source"
printf 'atoms\tmethod\texit_code\n' >"$output/attempts.tsv"
for case in 96:hf 96:pbe 96:r2scan 96:wb97m-v 300:hf 300:pbe; do
    atoms="${case%:*}"
    method="${case#*:}"
    name="oracle-${atoms}-${method}"
    code=0
    timeout --kill-after=15s 1200s "$python" -u "$root/endpoint.py" cpu "$method" direct "$atoms" \
        "$output/$name.json" --root "$root" >"$output/$name.txt" 2>&1 || code=$?
    printf '%s\t%s\t%d\n' "$atoms" "$method" "$code" >>"$output/attempts.tsv"
done

#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
output=$1
step=$2
sign=$3
convergence=$4
mkdir -p "$output"
scratch=$(mktemp -d /tmp/gqc3167-c7-fd.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
threads="${SLURM_CPUS_PER_TASK:?Missing Slurm CPUs per task}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$scratch"
{
    printf 'Slurm CPUs per task: %s\n' "$threads"
    lscpu
    df -h "$scratch"
    for f in /sys/fs/cgroup/{memory.max,memory.events,cpu.max}; do
        if [[ -r "$f" ]]; then printf '%s\n' "$f"; cat "$f"; fi
    done
} > "$output/hardware.txt"
"$GQC_PYTHON" -B -u "$root/c7_fd.py" --protocol "$root/protocol" \
    --output "$output" --scratch "$scratch" --threads "$threads" \
    --memory-gib 76 --step "$step" --sign "$sign" --convergence "$convergence" \
    > "$output/worker.log" 2>&1

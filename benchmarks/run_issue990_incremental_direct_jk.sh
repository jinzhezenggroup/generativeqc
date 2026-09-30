#!/usr/bin/env bash
# Matched #990 baseline/incremental qualification on the README direct-HF workload.
set -euo pipefail

: "${SLURM_JOB_ID:?Run inside a finite GPU Slurm allocation}"
: "${GENERATIVEQC_LIBRARY:?Select the Release CUDA library built from the PR source}"

python="${README_BENCHMARK_PYTHON:-python}"
output="${ISSUE990_BENCHMARK_OUTPUT:-.artifacts/issue990-incremental-direct-jk}"
point_timeout="${ISSUE990_POINT_TIMEOUT:-1800}"
rebuild_interval="${ISSUE990_REBUILD_INTERVAL:-8}"
[[ "$point_timeout" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid point timeout\n' >&2; exit 2; }
[[ "$rebuild_interval" =~ ^[0-9]+$ ]] || { printf 'Invalid rebuild interval\n' >&2; exit 2; }

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output"/{baseline,incremental}

run_point() {
    local mode="$1"
    local atoms="$2"
    local enabled=0
    [[ "$mode" == incremental ]] && enabled=1

    local prefix="$output/$mode/direct-$atoms"
    local env_args=(GENERATIVEQC_INCREMENTAL_DIRECT_JK="$enabled")
    if [[ "$mode" == incremental ]]; then
        env_args+=(GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL="$rebuild_interval")
    fi

    printf 'RUN mode=%s atoms=%s aos=%s\n' "$mode" "$atoms" "$((atoms * 8))"
    if timeout --kill-after=5 "$point_timeout"         env "${env_args[@]}" "$python" -m benchmarks.readme_hf_scaling         --case "water-$atoms" --batch 1 --repeats 3 --max-iterations 100         --energy-tolerance 1e-10 --density-tolerance 1e-9         --reference-gradient-tolerance 1e-8 --screening-tolerance 1e-12         --maximum-energy-error 1e-8 --maximum-force-error 1e-7         --output "$prefix.json" --progress-output "$prefix.progress.jsonl"         > "$prefix.log" 2>&1; then
        printf '{"exit_code":0,"time_limit_seconds":%s}\n' "$point_timeout" > "$prefix.outcome"
    else
        status=$?
        printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$status" "$point_timeout"             > "$prefix.outcome"
        printf 'FAIL mode=%s atoms=%s exit=%s (see %s.log)\n'             "$mode" "$atoms" "$status" "$prefix" >&2
        return "$status"
    fi
}

# Keep each baseline immediately adjacent to its incremental partner so the
# matched pair shares the same allocation, binary, device visibility, and local
# thermal/clock epoch. These are the #990 acceptance sizes: 96/192/384/768 AOs.
for atoms in 12 24 48 96; do
    run_point baseline "$atoms"
    run_point incremental "$atoms"
done

"$python" tools/reduce_issue990_incremental_direct_jk.py     --input "$output" --output "$output/summary.json"

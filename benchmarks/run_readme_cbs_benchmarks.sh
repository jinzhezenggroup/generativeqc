#!/usr/bin/env bash
# Each native point is independent of reference completion and later points.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun with an explicit finite --time}"
: "${QM7_DATASET:?Supply the pinned local qm7.mat}"
: "${README_CBS_SOURCE_REVISION:?Identify the measured production source}"
engine="${1:-native}"
case "$engine" in native|reference) ;; *) exit 2 ;; esac
if [[ "$engine" == native ]]; then
    : "${GENERATIVEQC_LIBRARY:?Select a source-matched Release native library}"
fi
python="${README_BENCHMARK_PYTHON:-python}"
output="${README_BENCHMARK_OUTPUT:-.artifacts/readme-cbs}"
limit="${README_BENCHMARK_POINT_TIMEOUT:-7200}"
samples="${README_BENCHMARK_SAMPLES:-1}"
[[ "$limit" =~ ^[1-9][0-9]*$ && "$samples" =~ ^[1-9][0-9]*$ ]] || exit 2
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -c 'from benchmarks._retention import raw_output_path; import sys; raw_output_path(sys.argv[1])' "$output"
failed=0
for carbon in ${README_BENCHMARK_SIZES:-1 2 3 4 5 6 7}; do
    directory="$output/$carbon"
    mkdir -p "$directory"
    result=0
    timeout --kill-after=15 "$limit" "$python" -m benchmarks.readme_cbs "$engine" \
        --qm7 "$QM7_DATASET" --carbon "$carbon" --samples "$samples" \
        --source-revision "$README_CBS_SOURCE_REVISION" \
        --hf-reference "${README_CBS_HF_REFERENCE:-exact}" \
        --policy "${README_CBS_POLICY:-standard}" \
        --budget-mib "${README_CBS_BUDGET_MIB:-65536}" --output "$directory/$engine.json" \
        > "$directory/$engine.log" 2>&1 || result=$?
    printf '{"exit_code":%s,"point_time_limit_seconds":%s}\n' "$result" "$limit" \
        > "$directory/$engine.outcome"
    printf 'C%s %s exit=%s\n' "$carbon" "$engine" "$result"
    [[ "$result" == 0 ]] || failed=1
done
exit "$failed"

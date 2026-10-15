#!/usr/bin/env bash
# Run from the recorded source checkout, inside one finite Slurm allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Use srun with --gres=gpu:5090:1 and an explicit finite --time}"
: "${GENERATIVEQC_LIBRARY:?Select the recorded Release/sm_120 native build}"
output="${1:?Supply a fresh ignored output directory}"
modes="${README_BENCHMARK_MODES:-none cuda}"
sizes="${README_BENCHMARK_SIZES:-3 12 48 96}"
export README_BENCHMARK_METHODS=wb97m-v
export README_BENCHMARK_REPEATS=5
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
failed=0
for mode in $modes; do
    export README_BENCHMARK_DENSITY_FITTING="$mode"
    export README_BENCHMARK_OUTPUT="$output/compiler-cache-priming/$mode"
    export README_BENCHMARK_SIZES="3 12"
    export README_BENCHMARK_POINT_TIMEOUT=1800
    bash benchmarks/run_readme_dft_benchmarks.sh native || exit 1
    date --iso-8601=ns > "$README_BENCHMARK_OUTPUT/completed.txt"
    export README_BENCHMARK_OUTPUT="$output/$mode"
    # Missing independent reference files never suppress native endpoints.
    # Restore matching reference/accuracy files here before accuracy replay.
    for atoms in $sizes; do
        [[ "$mode" != cuda || "$atoms" != 96 ]] || continue
        export README_BENCHMARK_SIZES="$atoms"
        export README_BENCHMARK_POINT_TIMEOUT=1800
        [[ "$atoms" != 96 ]] || export README_BENCHMARK_POINT_TIMEOUT=3600
        bash benchmarks/run_readme_dft_benchmarks.sh native || failed=1
    done
done
exit "$failed"

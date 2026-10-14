#!/usr/bin/env bash
# Run from the recorded source checkout, inside one finite Slurm allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Use srun with --gres=gpu:5090:1 and an explicit finite --time}"
: "${GENERATIVEQC_LIBRARY:?Select the recorded Release/sm_120 native build}"
output="${1:?Supply a fresh ignored output directory}"
export README_BENCHMARK_METHODS=wb97m-v
export README_BENCHMARK_REPEATS=5
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
failed=0
for mode in none cuda; do
    export README_BENCHMARK_DENSITY_FITTING="$mode"
    export README_BENCHMARK_OUTPUT="$output/$mode"
    # Missing independent reference files never suppress native endpoints.
    # Restore matching reference/accuracy files here before accuracy replay.
    for size_group in "3 12 48" "96"; do
        [[ "$mode" != cuda || "$size_group" != 96 ]] || continue
        export README_BENCHMARK_SIZES="$size_group"
        export README_BENCHMARK_POINT_TIMEOUT=1800
        [[ "$size_group" != 96 ]] || export README_BENCHMARK_POINT_TIMEOUT=3600
        bash benchmarks/run_readme_dft_benchmarks.sh native || failed=1
    done
done
exit "$failed"

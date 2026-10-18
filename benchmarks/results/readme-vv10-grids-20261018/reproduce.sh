#!/usr/bin/env bash
# Invoke from the repository root inside an explicitly finite GPU allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Use srun with an explicit finite --time}"
: "${GENERATIVEQC_LIBRARY:?Select a source-matched Release library}"
root="${README_GRID_OUTPUT:-.artifacts/readme-vv10-grids}"
export README_BENCHMARK_METHODS=wb97m-v
export README_BENCHMARK_SIZES="${README_BENCHMARK_SIZES:-3 12}"
export README_BENCHMARK_POINT_TIMEOUT="${README_BENCHMARK_POINT_TIMEOUT:-1200}"
export README_BENCHMARK_REPEATS=1
export README_BENCHMARK_GRIDS='24,8,16 32,12,24 32,16,32 40,16,32 48,12,24 48,16,32 64,12,24 64,16,32'
export README_BENCHMARK_DENSITY_FITTING=none
export README_BENCHMARK_OUTPUT="$root/reference-sweep"
bash benchmarks/run_readme_dft_benchmarks.sh reference
export README_BENCHMARK_REPEATS=5
export README_BENCHMARK_GRIDS='48,12,24 48,16,32'
for mode in none cuda; do
    export README_BENCHMARK_DENSITY_FITTING="$mode"
    export README_BENCHMARK_REPEATS=1
    export README_BENCHMARK_OUTPUT="$root/priming/$mode"
    bash benchmarks/run_readme_dft_benchmarks.sh both
    export README_BENCHMARK_REPEATS=5
    export README_BENCHMARK_OUTPUT="$root/measurements/$mode"
    bash benchmarks/run_readme_dft_benchmarks.sh both
done
python="${README_BENCHMARK_PYTHON:-python}"
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
"$python" -m tools.compare_readme_grids --raw "$root/reference-sweep" \
    --grids 24,8,16 32,12,24 32,16,32 40,16,32 48,12,24 48,16,32 64,12,24 64,16,32 \
    --baseline-grid 64,16,32 --sizes $README_BENCHMARK_SIZES \
    --energy-tolerance 1e-5 --force-tolerance 1e-4 --output "$root/report"
"$python" -m tools.render_readme_grids --raw "$root/measurements" \
    --grids 48,12,24 48,16,32 --sizes $README_BENCHMARK_SIZES --output "$root/report"

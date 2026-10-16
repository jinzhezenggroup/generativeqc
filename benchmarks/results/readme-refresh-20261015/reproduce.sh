#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use a finite Slurm GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?Preserve scheduler-assigned device visibility}"
root="${README_REFRESH_ROOT:?Select the immutable campaign directory}"
method="${1:?Select one README method}"
mode="${2:?Select none or cuda density fitting}"
case "$mode" in none|cuda) ;; *) exit 2 ;; esac
cd "$root/source"
export PATH="/data/jzzeng/qc-provider-1886-tools/bin:/home/jzzeng/codes/wb97m-20261002/ccache/usr/bin:/home/jzzeng/.local/bin:/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$root/runtime:/home/jzzeng/codes/wb97m-20261002/ccache/usr/lib/x86_64-linux-gnu:$CUDA_PATH/lib64"
export LD_PRELOAD="$CUDA_PATH/lib64/libcudart.so.12:$CUDA_PATH/lib64/libcublas.so.12:$CUDA_PATH/lib64/libcusolver.so.11"
export GENERATIVEQC_LIBRARY="$root/runtime/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$root/stationary-cache"
export PYTHONPATH=.:python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export README_BENCHMARK_METHODS="$method" README_BENCHMARK_REPEATS=5
export README_BENCHMARK_DENSITY_FITTING="$mode"
mkdir -p "$root/receipts"
date --iso-8601=ns
scontrol show job "$SLURM_JOB_ID" -o | tee "$root/receipts/$SLURM_JOB_ID.job.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv
ccache --version
if command -v sccache >/dev/null; then sccache --version; fi
ccache --show-stats
sha256sum "$GENERATIVEQC_LIBRARY"
failed=0
export README_BENCHMARK_POINT_TIMEOUT=1800
export README_BENCHMARK_OUTPUT="$root/priming/$mode"
export README_BENCHMARK_SIZES="3 12"
bash benchmarks/run_readme_dft_benchmarks.sh native || failed=1
date --iso-8601=ns > "$root/priming/$mode/$method/completed.txt"
export README_BENCHMARK_OUTPUT="$root/measurements/$mode"
for atoms in ${README_REFRESH_SIZES:-96 48 12 3}; do
    [[ "$method" != wb97m-v || "$mode" != cuda || "$atoms" != 96 ]] || continue
    directory="$README_BENCHMARK_OUTPUT/$method/$atoms"
    test ! -e "$directory/native.json"
    export README_BENCHMARK_SIZES="$atoms" README_BENCHMARK_POINT_TIMEOUT=1800
    if [[ "$method" == wb97m-v && "$atoms" == 96 ]]; then
        export README_BENCHMARK_POINT_TIMEOUT=3600
    fi
    bash benchmarks/run_readme_dft_benchmarks.sh native || failed=1
done
ccache --show-stats
date --iso-8601=ns
exit "$failed"

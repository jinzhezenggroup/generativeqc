#!/usr/bin/env bash
# Extend the frozen README cohort without rerunning or replacing its references.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun with an explicit finite --time}"
: "${CUDA_VISIBLE_DEVICES:?Preserve scheduler-assigned device visibility}"
root="${README_REFRESH_ROOT:-/data/jzzeng/gqc-readme-2531-2532-20261015-1593d3b2f}"
output="${README_DF96_OUTPUT:?Select a fresh supplement output directory}"
consumer="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/df96-benchmark-consumer.py.xz"
mkdir -p "$output"
test ! -e "$output/source"
cp -a "$root/source" "$output/source"
python3 - "$consumer" "$output/source/benchmarks/readme_omol25.py" <<'PY'
import lzma
import sys
from pathlib import Path

Path(sys.argv[2]).write_bytes(lzma.decompress(Path(sys.argv[1]).read_bytes()))
PY
cd "$output/source"
export PATH="/data/jzzeng/qc-provider-1886-tools/bin:/home/jzzeng/codes/wb97m-20261002/ccache/usr/bin:/home/jzzeng/.local/bin:/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$root/runtime:/home/jzzeng/codes/wb97m-20261002/ccache/usr/lib/x86_64-linux-gnu:$CUDA_PATH/lib64"
export LD_PRELOAD="$CUDA_PATH/lib64/libcudart.so.12:$CUDA_PATH/lib64/libcublas.so.12:$CUDA_PATH/lib64/libcusolver.so.11"
export GENERATIVEQC_LIBRARY="$root/runtime/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$root/stationary-cache"
export GENERATIVEQC_KS_RESIDENT_MAX_BYTES=2147483648
export PYTHONPATH=.:python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export README_BENCHMARK_POINT_TIMEOUT="${README_BENCHMARK_POINT_TIMEOUT:-14400}"
directory="$output/measurements/cuda/wb97m-v/96"
test ! -e "$directory/native.json"
mkdir -p "$output/receipts" "$directory"
date --iso-8601=ns | tee "$output/receipts/started.txt"
scontrol show job "$SLURM_JOB_ID" -o | tee "$output/receipts/slurm.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" | tee "$output/receipts/visibility.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv \
    | tee "$output/receipts/device.txt"
sccache --version | tee "$output/receipts/sccache.txt"
ccache --version | head -1 | tee "$output/receipts/ccache.txt"
sha256sum "$GENERATIVEQC_LIBRARY" | tee "$output/receipts/library.sha256"
sha256sum benchmarks/readme_omol25.py | tee "$output/receipts/consumer.sha256"
ccache --show-stats > "$output/receipts/ccache-before.txt"
result=0
timeout --kill-after=10 "$README_BENCHMARK_POINT_TIMEOUT" python -m benchmarks.readme_dft \
    --method wb97m-v native --atoms 96 --repeats 5 --density-fitting cuda \
    --basis-file "$root/measurements/cuda/inputs/def2-svp.json" \
    --auxiliary-basis-file benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json \
    --df-memory-budget-bytes 12884901888 \
    --force-max-device-bytes 4294967296 --force-max-host-bytes 4294967296 \
    --allow-missing-reference --output "$directory/native.json" \
    > "$directory/native.log" 2>&1 || result=$?
printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$result" "$README_BENCHMARK_POINT_TIMEOUT" \
    > "$directory/native.outcome"
printf '%s\n' "$result" > "$output/receipts/exit-code.txt"
ccache --show-stats > "$output/receipts/ccache-after.txt"
date --iso-8601=ns | tee "$output/receipts/completed.txt"
exit "$result"

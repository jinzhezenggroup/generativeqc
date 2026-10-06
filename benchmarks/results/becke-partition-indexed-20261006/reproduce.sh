#!/usr/bin/env bash
set -euo pipefail

# Restore publication.source.revision and apply source.patch before reproducing.
# The finite step is mandatory; never change Slurm-assigned device visibility.
if [[ -z ${SLURM_JOB_ID:-} || ${SLURM_STEP_ID:-batch} == batch || ${SLURM_STEP_ID:-} == extern ]]; then
  exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --time=00:15:00 bash "$0" "$@"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm-assigned device visibility}"
bundle=$(cd "$(dirname "$0")" && pwd)
root=$(git rev-parse --show-toplevel)
cd "$root"
sha256sum -c "$bundle/source-files.sha256"
python=${PYTHON:-python}
cuda=${CUDA_ROOT:-/group/software/cuda-12.9.1}
work=${1:-$root/.artifacts/benchmarks/becke-partition-indexed-$(date +%Y%m%dT%H%M%S)}
mkdir -p "$work/compiler"
export PATH="$cuda/bin:$PATH" PYTHONPATH="$root/python:$root"
export LD_LIBRARY_PATH="$cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCACHE_BASEDIR="$root" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ccache --version > "$work/ccache-version.txt"
ccache --show-stats > "$work/ccache-before.txt"
printf '%s\n' '#!/bin/sh' 'exec ccache /usr/bin/c++ "$@"' > "$work/compiler/c++"
chmod +x "$work/compiler/c++"
export CXX="$work/compiler/c++"
"$python" -m benchmarks.becke_phased_probe --partition-derivative --emit "$work/partition.cu"
ccache nvcc -std=c++20 -O3 --expt-relaxed-constexpr --fmad=false \
  -arch=sm_120 -Xcompiler=-fPIC -shared "$work/partition.cu" -o "$work/partition.so"
ccache --show-stats > "$work/ccache-after.txt"
export GENERATIVEQC_BECKE_PARTITION_PROBE="$work/partition.so"
scontrol show step "$SLURM_JOB_ID.$SLURM_STEP_ID" -o > "$work/step.txt"
"$python" -m pytest -q tests/python/test_becke_partition_cuda.py > "$work/cuda-tests.log" 2>&1
for tool in memcheck initcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 9 \
    "$python" -m benchmarks.becke_phased_probe --partition-derivative \
    --library "$work/partition.so" --atoms 48 96 --tiles 2 --tile-points 17 256 \
    --output "$work/$tool.json" > "$work/$tool.log" 2>&1
done
# Sanitizer timings are intrusive; only this separate pass is performance data.
"$python" -m benchmarks.becke_phased_probe --partition-derivative \
  --library "$work/partition.so" --atoms 48 96 --tiles 64 --tile-points 256 1024 \
  --output "$work/isolated-abba.json" > "$work/isolated-abba.log" 2>&1

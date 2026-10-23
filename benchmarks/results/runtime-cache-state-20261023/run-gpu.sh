#!/usr/bin/env bash
# One physical Slurm GPU covers the separate pilot and all 30 formal processes.
set -euo pipefail
root=/data/srxu/projects/gqc-runtime-cache-state-20261023
cd "$root"
: "${SLURM_JOB_ID:?requires real Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?requires actual Slurm-assigned GPU visibility}"
test -f receipts/sdk-ready
test ! -e receipts/gpu-job-started
printf '%s\n' "$SLURM_JOB_ID" > receipts/gpu-job-started
trap 'rc=$?; printf "%s\n" "$rc" > receipts/gpu-job-exit-code' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
/usr/bin/scontrol show job "$SLURM_JOB_ID" > receipts/gpu-allocation.txt
export PATH="$root/runtime-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export PYTHONPATH="$root/source/python:$root/harness"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum --check receipts/frozen-config.sha256
sha256sum --check receipts/measurement-driver.sha256
python -m benchmarks.runtime_cache_state run --config "$root/frozen-inputs/config.json" \
  --output "$root/pilot" --pilot --timeout 7200
python -m benchmarks.runtime_cache_state run --config "$root/frozen-inputs/config.json" \
  --output "$root/formal" --pilot-receipt "$root/pilot" --timeout 7200
python -m benchmarks.runtime_cache_state qualify --campaign "$root/formal" \
  --output "$root/formal-independent-replay.json"
python "$root/freeze_environment.py" --output "$root/receipts/scientific-environment-after.json"
cmp receipts/scientific-environment.json receipts/scientific-environment-after.json
printf 'measured plus independently replayed; publication/review/merge remain separate\n' > receipts/measurement-finished

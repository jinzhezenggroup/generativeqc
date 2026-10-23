#!/usr/bin/env bash
# Submit only with afterok dependency on the actual native build job.
set -euo pipefail
root=/data/srxu/projects/gqc-runtime-cache-state-20261023
cd "$root"
: "${SLURM_JOB_ID:?requires real CPU Slurm allocation}"
test ! -e receipts/freeze-job-started
printf '%s\n' "$SLURM_JOB_ID" > receipts/freeze-job-started
trap 'rc=$?; printf "%s\n" "$rc" > receipts/freeze-job-exit-code' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
test -f receipts/build-complete
test "$(cat receipts/resume-exit-code)" = 0
/usr/bin/scontrol show job "$SLURM_JOB_ID" > receipts/freeze-allocation.txt
export PATH="$root/runtime-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export PYTHONPATH="$root/source/python:$root/harness"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$root/freeze_environment.py" --output "$root/receipts/scientific-environment-recheck.json"
cmp receipts/scientific-environment.json receipts/scientific-environment-recheck.json
python "$root/harness/benchmarks/results/runtime-cache-state-20261023/freeze_sdk.py" \
  --root "$root" --qm7 "$root/qm7.mat"
sha256sum frozen-inputs/config.json > receipts/frozen-config.sha256
printf 'CPU SDK/source/AOT/environment verified; GPU NOT_RUN\n' > receipts/sdk-ready

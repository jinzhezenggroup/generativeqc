#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ragged-default-20261012
export PATH="/data/jzzeng/qc-ace-master-20261012/venv311/bin:/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="$root/source/build-cuda:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
printf 'Slurm job=%s visible=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
cd "$root/source"
build-cuda/generativeqc_dft_cuda_tests --point-batches
build-cuda/generativeqc_dft_cuda_tests --ragged-batches
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --log-file "$root/validation/memcheck.log" \
  build-cuda/generativeqc_dft_cuda_tests --ragged-batches
compute-sanitizer --tool racecheck --error-exitcode 1 \
  --log-file "$root/validation/racecheck.log" \
  build-cuda/generativeqc_dft_cuda_tests --ragged-batches

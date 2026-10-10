#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
out="$root/auto-15e052d8/memcheck"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" > "$out/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=uuid,name,driver_version,memory.total,power.limit --format=csv > "$out/gpu.csv"
sha256sum "$root/source/build-cuda/libgenerativeqc.so" "$out/probe" \
  "$out/methane-aug-cc-pvtz.input" > "$out/inputs.sha256"
GENERATIVEQC_DF_PROGRESS_TRACE="$out/progress.jsonl" \
  compute-sanitizer --tool memcheck --error-exitcode=99 --leak-check=full \
    "$out/probe" "$out/methane-aug-cc-pvtz.input" > "$out/result.log" 2>&1
printf '%s\n' complete > "$out/status.txt"

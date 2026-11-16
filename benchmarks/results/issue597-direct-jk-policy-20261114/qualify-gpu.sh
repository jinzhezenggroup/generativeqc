#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd /data/jzzeng/issue597-20261114
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/candidate:$PWD/blas:/group/software/cuda-12.9.1/lib64"
scontrol -d show job "$SLURM_JOB_ID" > evidence/qualification-allocation.txt
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" > evidence/qualification-visibility.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.limit --format=csv > evidence/gpu-before.csv
sha256sum baseline/libgenerativeqc.so.0.1.0 candidate/libgenerativeqc.so.0.1.0 baseline/generativeqc_direct_jk_policy_benchmark candidate/generativeqc_direct_jk_policy_benchmark > evidence/artifact-sha256.txt
candidate/generativeqc_hf_matrix_profile_cuda_tests > evidence/candidate-oracle-final.log 2>&1
nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv --loop-ms=50 > evidence/process-memory.csv &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
for run in 0 1; do
  if (( run % 2 == 0 )); then
    variants='baseline candidate'
  else
    variants='candidate baseline'
  fi
  for variant in $variants; do
    export LD_LIBRARY_PATH="$PWD/$variant:$PWD/blas:/group/software/cuda-12.9.1/lib64"
    "$variant/generativeqc_direct_jk_policy_benchmark" 3 > "evidence/$variant-$run.jsonl" 2> "evidence/$variant-$run.err" &
    endpoint=$!
    printf '%s\t%s\t%s\n' "$variant" "$run" "$endpoint" >> evidence/endpoint-pids.tsv
    wait "$endpoint"
  done
done
kill "$sampler"
wait "$sampler" 2>/dev/null || true
trap - EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.limit --format=csv > evidence/gpu-after.csv
export LD_LIBRARY_PATH="$PWD/candidate:$PWD/blas:/group/software/cuda-12.9.1/lib64"
timeout 180s /group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 candidate/generativeqc_direct_persistent_queue_cuda_tests > evidence/queue-memcheck.log 2>&1
timeout 180s /group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 candidate/generativeqc_hf_matrix_profile_cuda_tests --capacity-only > evidence/capacity-memcheck.log 2>&1

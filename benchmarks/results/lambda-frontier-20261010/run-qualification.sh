#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
campaign="$root/p2-lambda-frontier-master"
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$campaign/candidate:$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
bash "$campaign/run-gpu-master.sh"
bash "$campaign/run-gpu-master.sh" memcheck
out="$campaign/profile-$SLURM_JOB_ID"
mkdir -p "$out"
cp "$0" "$out/reproduce.sh"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "HOSTNAME=$(hostname)" > "$out/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,memory.total,power.limit --format=csv > "$out/gpu.csv"
sha256sum "$campaign/candidate/endpoint" "$campaign/candidate/libgenerativeqc.so" > "$out/binaries.sha256"
controls=(1 1 1 1 32 8 8 0 1 2 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1)
env -u GENERATIVEQC_RHF_RESIDENT_VALUES \
  GENERATIVEQC_DF_PROGRESS_TRACE="$out/candidate.progress.jsonl" \
  GENERATIVEQC_DF_TRACE="$out/candidate.components.jsonl" \
  /usr/bin/time -v -o "$out/candidate.time.txt" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --cuda-event-trace=false --cuda-memory-usage=true \
    --force-overwrite=true --output="$out/candidate" \
    "$campaign/candidate/endpoint" "$root/ethane230.input" "$out/candidate.json" \
    "${controls[@]}" > "$out/candidate.log" 2>&1
nsys export --type=sqlite --output="$out/candidate.sqlite" "$out/candidate.nsys-rep" > "$out/export.log" 2>&1
printf '%s\n' complete > "$out/status.txt"

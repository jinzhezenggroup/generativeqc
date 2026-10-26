#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/issue2516-20261027
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$root/source/build-cuda:$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
out="$root/evidence/profile-$SLURM_JOB_ID"
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "HOSTNAME=$(hostname)" > "$out/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,power.limit --format=csv > "$out/gpu.csv"
nsys --version > "$out/nsys-version.txt"
controls=(1 1 1 1 32 8 8 0 1 auto 1 30 0 0 1 auto 1 0 auto auto 0 30 1 1 1 1 0 0 0)
printf '%s\n' "${controls[*]}" > "$out/controls.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 200 > "$out/memory.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
/usr/bin/time -v -o "$out/ethane230.time.txt" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --cuda-event-trace=false --cuda-memory-usage=true \
  --force-overwrite=false --output="$out/profile" \
  "$root/evidence/endpoint" "$root/evidence/ethane230.input" "$out/ethane230.json" "${controls[@]}" > "$out/profile.log" 2>&1
nsys export --type=sqlite --output="$out/profile.sqlite" "$out/profile.nsys-rep" > "$out/export.log" 2>&1

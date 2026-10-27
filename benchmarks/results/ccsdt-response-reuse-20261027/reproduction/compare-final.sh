#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/issue2516-20261027
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
out="$root/evidence/compare-final-$SLURM_JOB_ID"
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "HOSTNAME=$(hostname)" > "$out/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,power.limit --format=csv > "$out/gpu.csv"
controls=(1 1 1 1 32 8 8 0 1 auto 1 30 0 0 1 auto 1 0 auto auto 0 30 1 1 1 1 0 0 0)
printf '%s\n' "${controls[*]}" > "$out/controls.txt"
index=0
for variant in qualified control control qualified; do
  index=$((index+1))
  export LD_LIBRARY_PATH="$root/evidence/$variant:$deps/lib:/group/software/cuda-12.9.1/lib64"
  ldd "$root/evidence/$variant/endpoint" > "$out/$index-$variant.loader.txt"
  grep -F "$root/evidence/$variant/libgenerativeqc.so.0" "$out/$index-$variant.loader.txt" > /dev/null
  sha256sum "$root/evidence/$variant/endpoint" "$root/evidence/$variant/libgenerativeqc.so.0" > "$out/$index-$variant.binaries.sha256"
  /usr/bin/time -v -o "$out/$index-$variant.time.txt" \
    "$root/evidence/$variant/endpoint" "$root/evidence/ethane230.input" "$out/$index-$variant.json" "${controls[@]}" > "$out/$index-$variant.log" 2>&1
done

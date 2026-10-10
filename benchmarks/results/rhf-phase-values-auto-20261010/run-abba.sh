#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$root/source"
out="$root/auto-15e052d8/abba-$SLURM_JOB_ID"
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" > "$out/allocation.txt"
cp "$0" "$out/reproduce.sh"
cp "$root/build-auto/binaries.sha256" "$out/"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=uuid,name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory \
  --format=csv > "$out/gpu.csv"
sha256sum "$root/ethane230.input" > "$out/input.sha256"
controls=(1 1 1 1 32 8 8 0 1 2 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1)
printf '%s\n' "${controls[*]}" > "$out/controls.txt"
for index in 1 2 3 4; do
  case "$index" in 1|4) name=baseline ;; *) name=candidate ;; esac
  sample="$name-$index"
  printf '%s %s started\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
  nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv --loop-ms=1000 > "$out/$sample.gpu.csv" &
  sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true' EXIT
  environment=(env -u GENERATIVEQC_DF_TRACE -u GENERATIVEQC_DF_PROGRESS_TRACE \
    -u GENERATIVEQC_RHF_RESIDENT_VALUES)
  if [[ "$name" == baseline ]]; then
    environment+=(GENERATIVEQC_RHF_RESIDENT_VALUES=0)
  fi
  "${environment[@]}" /usr/bin/time -v -o "$out/$sample.time.txt" \
    "$root/build-auto/endpoint" "$root/ethane230.input" "$out/$sample.json" "${controls[@]}" \
    > "$out/$sample.log" 2>&1
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  trap - EXIT
  printf '%s %s complete\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
done
printf '%s\n' complete > "$out/status.txt"

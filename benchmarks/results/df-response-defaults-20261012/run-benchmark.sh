#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-cc-response-defaults-20261012-2a6de26e
out="$root/evidence/defaults-$SLURM_JOB_ID"
mkdir -p "$out"
export PATH="/data/jzzeng/qc-rhf-shell-bilinear-master-20261012-531364e1/venv/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$root/default/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
cp "$0" "$out/reproduce.sh"
cp "$root/default/source.sha256" "$root/default/binaries.sha256" "$out/"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" "START=$(date --iso-8601=seconds)" > "$out/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,memory.total,power.limit \
  --format=csv > "$out/gpu.csv"
index=0
for side in baseline candidate candidate baseline baseline candidate candidate baseline baseline candidate; do
  index=$((index+1))
  sample="$side-$index"
  controls=(1 1 1 1 32 8 0 0 1 auto 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1 0 0 0 0)
  if [[ "$side" == candidate ]]; then controls[19]=auto; controls[25]=auto; fi
  printf '%s %s started\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
  printf '%s\n' "${controls[*]}" > "$out/$sample.controls.txt"
  nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv --loop-ms=1000 > "$out/$sample.gpu.csv" &
  sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true' EXIT
  env -u GENERATIVEQC_DF_TRACE -u GENERATIVEQC_DF_PROGRESS_TRACE \
    -u GENERATIVEQC_RHF_RESIDENT_VALUES \
    /usr/bin/time -v -o "$out/$sample.time.txt" "$root/default/bin/endpoint" \
    "$root/input/ethane230.input" "$out/$sample.json" "${controls[@]}" \
    > "$out/$sample.log" 2>&1
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  trap - EXIT
  python "$root/check-result.py" "$out/$sample.json" "$side" > "$out/$sample.gates.json"
  printf '%s %s complete\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
done
printf '%s\n' complete > "$out/status.txt"

#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-rhf-shell-bilinear-master-20261012-531364e1
out="$root/evidence/other-$SLURM_JOB_ID"
mkdir -p "$out"
cd "$root/source"
export PATH="$root/venv/bin:/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="$root/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" "START=$(date --iso-8601=seconds)" > "$out/allocation.txt"
cp "$0" "$out/reproduce.sh"
cp "$root/evidence/binaries.sha256" "$out/"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=uuid,name,driver_version,memory.total,power.limit \
  --format=csv > "$out/gpu.csv"
input="$PWD/benchmarks/results/rhf-phase-values-auto-20261010/ethane230.input"
sha256sum "$input" > "$out/input.sha256"
for cohort in scalar-fusion scratch-reuse; do
  for index in 1 2 3 4; do
    case "$index" in 1|4) side=baseline ;; *) side=candidate ;; esac
    sample="$cohort-$side-$index"
    controls=(1 1 1 1 32 8 0 0 1 1 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1 0 0 0 0)
    if [[ "$cohort" == scalar-fusion && "$side" == candidate ]]; then controls[19]=1; fi
    if [[ "$cohort" == scratch-reuse && "$side" == candidate ]]; then controls[25]=1; fi
    printf '%s %s started\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
    printf '%s\n' "${controls[*]}" > "$out/$sample.controls.txt"
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
      --format=csv --loop-ms=1000 > "$out/$sample.gpu.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    env -u GENERATIVEQC_DF_TRACE -u GENERATIVEQC_DF_PROGRESS_TRACE \
      -u GENERATIVEQC_RHF_RESIDENT_VALUES \
      /usr/bin/time -v -o "$out/$sample.time.txt" \
      "$root/bin/endpoint" "$input" "$out/$sample.json" "${controls[@]}" \
      > "$out/$sample.log" 2>&1
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    trap - EXIT
    python "$root/evidence/check-result.py" "$out/$sample.json" derivative candidate \
      > "$out/$sample.gates.json"
    printf '%s %s complete\n' "$(date --iso-8601=seconds)" "$sample" >> "$out/progress.txt"
  done
done
printf '%s\n' complete > "$out/status.txt"

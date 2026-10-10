#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/source/python:$root/source:$deps"
export GENERATIVEQC_LIBRARY="$root/source/build-cuda/libgenerativeqc.so"
cd "$root/source"
out="$root/auto-15e052d8/qualification-$SLURM_JOB_ID"
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" > "$out/allocation.txt"
cp "$0" "$out/reproduce.sh"
cp "$root/build-auto/binaries.sha256" "$out/"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=uuid,name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory \
  --format=csv > "$out/gpu.csv"
GENERATIVEQC_RCCSD_CUDA_TEST=1 /home/jzzeng/miniconda3/bin/python -m pytest \
  tests/python/test_rhf_phase_resident_values_cuda.py \
  tests/python/test_rhf_reference_residency.py -q --tb=short \
  --basetemp="$out/pytest" > "$out/tests.log" 2>&1
controls=(1 1 1 1 32 8 8 0 1 2 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1)
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv --loop-ms=1000 > "$out/candidate.gpu.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
env -u GENERATIVEQC_RHF_RESIDENT_VALUES \
  GENERATIVEQC_DF_PROGRESS_TRACE="$out/candidate.progress.jsonl" \
  GENERATIVEQC_DF_TRACE="$out/candidate.components.jsonl" \
  /usr/bin/time -v -o "$out/candidate.time.txt" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --cuda-event-trace=false --cuda-memory-usage=true \
    --force-overwrite=true --output="$out/candidate" \
    "$root/build-auto/endpoint" "$root/ethane230.input" "$out/candidate.json" "${controls[@]}" \
    > "$out/candidate.log" 2>&1
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
nsys export --type=sqlite --output="$out/candidate.sqlite" "$out/candidate.nsys-rep" \
  > "$out/export.log" 2>&1
printf '%s\n' complete > "$out/status.txt"

#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$root/p1-force-class-domains/candidate:$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/source/python:$root/source:$deps"
export GENERATIVEQC_LIBRARY="$root/p1-force-class-domains/candidate/libgenerativeqc.so"
cd "$root/source"
out="$root/p1-force-class-domains/qualification-$SLURM_JOB_ID"
mkdir -p "$out"
printf '%s\n' "SLURM_JOB_ID=$SLURM_JOB_ID" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "HOSTNAME=$(hostname)" > "$out/allocation.txt"
cp "$0" "$out/reproduce.sh"
sha256sum "$root/p1-force-class-domains/candidate/libgenerativeqc.so" \
  "$root/p1-force-class-domains/candidate/endpoint" "$root/p1-force-class-domains/candidate/provider" \
  > "$out/binaries.sha256"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=uuid,name,driver_version,memory.total,power.limit --format=csv > "$out/gpu.csv"
if [[ "${P1_PROFILE_ONLY:-0}" != 1 ]]; then
GENERATIVEQC_RCCSDT_CUDA_TEST=1 /home/jzzeng/miniconda3/bin/python -m pytest \
  tests/python/test_direct_force_class_domains_cuda.py -q --tb=short \
  --basetemp="$out/pytest" > "$out/fci-tests.log" 2>&1
"$root/p1-force-class-domains/candidate/provider" --mixed-force-domains-only > "$out/provider.log" 2>&1
env GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES=0 \
  "$root/p1-force-class-domains/candidate/provider" --mixed-force-domains-only > "$out/provider-fallback.log" 2>&1
fi
if [[ "${P1_SKIP_MEMCHECK:-0}" != 1 ]]; then
compute-sanitizer --tool memcheck --error-exitcode=99 --leak-check=full --launch-timeout=600 \
  "$root/p1-force-class-domains/candidate/provider" --mixed-force-domains-only > "$out/memcheck.log" 2>&1
fi
if [[ "${P1_MEMCHECK_ONLY:-0}" == 1 ]]; then
  printf '%s\n' complete > "$out/status.txt"
  exit 0
fi
controls=(1 1 1 1 32 8 8 0 1 2 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1)
env -u GENERATIVEQC_RHF_RESIDENT_VALUES \
  GENERATIVEQC_DF_PROGRESS_TRACE="$out/candidate.progress.jsonl" \
  GENERATIVEQC_DF_TRACE="$out/candidate.components.jsonl" \
  /usr/bin/time -v -o "$out/candidate.time.txt" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --cuda-event-trace=false --cuda-memory-usage=true \
    --force-overwrite=true --output="$out/candidate" \
    "$root/p1-force-class-domains/candidate/endpoint" "$root/ethane230.input" "$out/candidate.json" \
    "${controls[@]}" > "$out/candidate.log" 2>&1
nsys export --type=sqlite --output="$out/candidate.sqlite" "$out/candidate.nsys-rep" > "$out/export.log" 2>&1
printf '%s\n' complete > "$out/status.txt"

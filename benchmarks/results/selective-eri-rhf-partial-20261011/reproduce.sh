#!/usr/bin/env bash
set -euo pipefail
# Build master and the retained reconstruction separately, with the same CUDA
# configuration. This script never changes Slurm visibility or loads two native
# providers into one Python process.
: "${BASE_ROOT:?immutable master checkout/build root}"
: "${CANDIDATE_ROOT:?reconstructed candidate checkout/build root}"
: "${CUDA_HOME:?CUDA 12.9.1 toolkit}"
: "${PYTHON:?Python with repository dependencies and PySCF}"
: "${DEPENDENCY_ROOT:?directory containing the PySCF/native dependency libraries}"
here=$(cd "$(dirname "$0")" && pwd)
for root in "$BASE_ROOT" "$CANDIDATE_ROOT"; do
  ccache c++ -std=c++20 -O2 -fPIC -Wall -Wextra -Werror \
    -I"$CANDIDATE_ROOT/include" -I"$CANDIDATE_ROOT/src" -I"$CUDA_HOME/include" \
    -c "$CANDIDATE_ROOT/benchmarks/selective_eri_rhf_endpoint.cpp" -o "$root/endpoint.o"
  c++ -shared "$root/endpoint.o" -L"$root/build-cuda" -L"$CUDA_HOME/lib64" \
    -lgenerativeqc -lcudart -ldl -Wl,-z,defs \
    -Wl,-rpath,"$root/build-cuda" -o "$root/endpoint.so"
done
export PYTHONPATH="$CANDIDATE_ROOT/python:$CANDIDATE_ROOT:$DEPENDENCY_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export GENERATIVEQC_RHF_RESIDENT_VALUES=auto
export OUTPUT_ROOT="$CANDIDATE_ROOT/.artifacts/selective-eri-reproduction"
mkdir -p "$OUTPUT_ROOT"
export INPUT="$OUTPUT_ROOT/basis.input"
gzip -dc "$here/basis.input.gz" > "$INPUT"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=02:00:00 \
  bash -s <<'GPU'
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm assignment}"
for repeat in 0 1 2; do
  if (( repeat % 2 == 0 )); then routes=(candidate baseline); else routes=(baseline candidate); fi
  for route in "${routes[@]}"; do
    if [[ $route == baseline ]]; then
      root="$BASE_ROOT"
      unset GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK GENERATIVEQC_RHF_RESIDENT_MAX_BYTES
    else
      root="$CANDIDATE_ROOT"
      export GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK=66058240
      export GENERATIVEQC_RHF_RESIDENT_MAX_BYTES=2147483648
    fi
    export GENERATIVEQC_LIBRARY="$root/build-cuda/libgenerativeqc.so"
    export LD_LIBRARY_PATH="$root/build-cuda:$DEPENDENCY_ROOT/pyscf/lib:$DEPENDENCY_ROOT/lib:$CUDA_HOME/lib64"
    export GENERATIVEQC_DF_PROGRESS_TRACE="$OUTPUT_ROOT/$route-$repeat.trace.jsonl"
    "$PYTHON" -m benchmarks.selective_eri_rhf_endpoint \
      --input "$INPUT" --probe "$root/endpoint.so" \
      --output "$OUTPUT_ROOT/$route-$repeat.json" --budget-bytes 4294967296
  done
done
GPU
"$PYTHON" -m benchmarks.selective_eri_rhf_acceptance oracle \
  --input "$INPUT" --output "$OUTPUT_ROOT/oracle.json"
"$PYTHON" -m benchmarks.selective_eri_rhf_acceptance compare \
  --baseline "$OUTPUT_ROOT"/baseline-{0,1,2}.json \
  --candidate "$OUTPUT_ROOT"/candidate-{0,1,2}.json \
  --oracle "$OUTPUT_ROOT/oracle.json" --output "$OUTPUT_ROOT/comparison.json"

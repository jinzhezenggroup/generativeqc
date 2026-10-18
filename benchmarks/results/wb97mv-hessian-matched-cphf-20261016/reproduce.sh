#!/usr/bin/env bash
set -euo pipefail

expected=17c539db5aa05dce9e2396aef814732f68cdfcb6
test "$(git rev-parse HEAD)" = "$expected" || {
  echo "Retain this script outside the checkout, then restore source $expected." >&2
  exit 2
}
cuda_root=${CUDA_ROOT:-/group/software/cuda-12.9.1}
compiler_python=${COMPILER_PYTHON:-python3}
reference_python=${REFERENCE_PYTHON:?Set REFERENCE_PYTHON to the GPU4PySCF environment}
build=${BUILD_DIRECTORY:-build/native-matched-hessian-reproduce}
mkdir -p "$build"
cache=$(command -v sccache || command -v ccache) || {
  echo "A working compiler cache is required." >&2
  exit 2
}
"$cache" --version
"$cache" --show-stats > "$build/compiler-cache-before.txt"
export CCACHE_BASEDIR=$PWD
cmake -S . -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$cache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache" \
  -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_BUILD_CLI=ON \
  -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON \
  -DGENERATIVEQC_HESSIAN_AOT_INVENTORY=
cmake --build "$build" --target generativeqc_cli
"$cache" --show-stats > "$build/compiler-cache-after.txt"
build=$(cd "$build" && pwd)
"$compiler_python" - "$build/generated/hessian_aot/inventory.json" <<'PY'
import hashlib
import sys
from pathlib import Path
observed = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
assert observed == 'c349bc9f63d543f5a1a33bd9ca2552a568d6846c9d5f4d83b5222d15a04770c6'
PY
output=${OUTPUT:-.artifacts/wb97mv-native-matched-reproduced.json}

salloc --exclusive --partition=main --nodelist=node2 --gres=gpu:pro6000:4 \
  --cpus-per-task=8 --mem=16G --nodes=1 --ntasks=1 --time=00:20:00 \
  srun --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --time=00:18:00 \
  env CUDA_PATH="$cuda_root" CUDA_HOME="$cuda_root" CUDA_DISABLE_PTX_JIT=1 \
  PATH="$cuda_root/bin:$PATH" \
  LD_LIBRARY_PATH="$build:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH=python:. GENERATIVEQC_LIBRARY="$build/libgenerativeqc.so.0.1.0" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$reference_python" -m benchmarks.compare_wb97mv_hessian_gpu4pyscf \
  --cli "$build/generativeqc" --reference-python "$reference_python" \
  --cases h2 water --repeats 2 --timeout-seconds 600 --source-commit "$expected" \
  --output "$output"

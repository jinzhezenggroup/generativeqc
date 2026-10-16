#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected=74c90701cfc95a2f7faa0f8e0397692197180a6a
test "$(git rev-parse HEAD)" = "$expected" || {
  echo "Copy the bundle outside the checkout, then restore source $expected." >&2
  exit 2
}
cuda_root=${CUDA_ROOT:-/group/software/cuda-12.9.1}
reference_python=${REFERENCE_PYTHON:?Set REFERENCE_PYTHON to the GPU4PySCF 1.8.1 environment}
build=${BUILD_DIRECTORY:-build/native-hessian-reproduce}
mkdir -p "$build"
gzip -dc "$bundle/inventory.json.gz" > "$build/hessian-inventory.json"
inventory=$(cd "$build" && pwd)/hessian-inventory.json
cache=$(command -v sccache || command -v ccache) || {
  echo "A working compiler cache is required." >&2
  exit 2
}
"$cache" --version
export CCACHE_BASEDIR=$PWD
cmake -S . -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$cache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache" \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_BUILD_CLI=ON \
  -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON \
  -DGENERATIVEQC_HESSIAN_AOT_INVENTORY="$inventory"
cmake --build "$build" --target generativeqc_cli
build=$(cd "$build" && pwd)
output=${OUTPUT:-.artifacts/wb97mv-native-parallel-reproduced.json}

salloc --exclusive --partition=main --nodelist=node2 --gres=gpu:pro6000:4 \
  --cpus-per-task=4 --mem=8G --nodes=1 --ntasks=1 --time=00:15:00 \
  srun --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  env CUDA_PATH="$cuda_root" CUDA_HOME="$cuda_root" \
  PATH="$cuda_root/bin:$PATH" \
  LD_LIBRARY_PATH="$build:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH=python:. GENERATIVEQC_LIBRARY="$build/libgenerativeqc.so.0.1.0" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$reference_python" -m benchmarks.compare_wb97mv_hessian_gpu4pyscf \
  --cli "$build/generativeqc" --reference-python "$reference_python" \
  --cases h2 water --repeats 2 --timeout-seconds 600 --source-commit "$expected" \
  --output "$output"

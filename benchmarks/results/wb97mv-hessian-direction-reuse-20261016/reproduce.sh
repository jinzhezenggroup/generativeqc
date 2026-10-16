#!/usr/bin/env bash
set -euo pipefail

: "${BASELINE_CHECKOUT:?Set to a clean checkout of f37f96a1c3fee0da04ea0cc2c490f04140d65ae6}"
: "${CANDIDATE_CHECKOUT:?Set to a clean checkout of 5aa703721fdce6a220838676af615d574f0fed05}"
: "${REFERENCE_PYTHON:?Set to the pinned GPU4PySCF/PySCF/CuPy environment}"
bundle=$(cd "$(dirname "$0")" && pwd)
driver_directory=$(mktemp -d "${TMPDIR:-/tmp}/wb97mv-ab.XXXXXX")
trap 'rm -rf -- "$driver_directory"' EXIT
xz -dc "$bundle/compare-native-ab.py.xz" > "$driver_directory/compare-native-ab.py"
cuda_root=${CUDA_ROOT:-/group/software/cuda-12.9.1}
cache=$(command -v sccache || command -v ccache) || {
  echo 'A working compiler cache is required.' >&2
  exit 2
}
"$cache" --version
for arm in baseline candidate; do
  if [[ "$arm" == baseline ]]; then
    checkout=$BASELINE_CHECKOUT
    expected=f37f96a1c3fee0da04ea0cc2c490f04140d65ae6
  else
    checkout=$CANDIDATE_CHECKOUT
    expected=5aa703721fdce6a220838676af615d574f0fed05
  fi
  test "$(git -C "$checkout" rev-parse HEAD)" = "$expected"
  test -z "$(git -C "$checkout" status --porcelain)"
  checkout=$(cd "$checkout" && pwd)
  build="$checkout/build/native-hessian-optimized-reproduce"
  mkdir -p "$build"
  "$cache" --show-stats > "$build/cache-before.txt"
  export CCACHE_BASEDIR=$checkout
  cmake -S "$checkout" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/home/jzzeng/miniconda3}" \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$cache" -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache" \
    -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120 \
    -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 \
    -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF -DGENERATIVEQC_ENABLE_CUDA=ON \
    -DGENERATIVEQC_BUILD_CLI=ON -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON \
    -DGENERATIVEQC_HESSIAN_AOT_INVENTORY=
  cmake --build "$build" --target generativeqc_cli --parallel "${BUILD_JOBS:-6}" --verbose
  "$cache" --show-stats > "$build/cache-after.txt"
  if [[ "$arm" == baseline ]]; then baseline=$build; else candidate=$build; fi
done
cd "$CANDIDATE_CHECKOUT"
output=${OUTPUT:-.artifacts/optimized-hessian-ab-reproduced.json}

# Run on the node2 Slurm submission host, or adapt node/GRES together. Preserve
# Slurm's single-device visibility in the nested benchmark step.
salloc --exclusive --partition=main --nodelist=node2 --gres=gpu:pro6000:4 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00:20:00 \
  srun --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:18:00 \
  env CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root" CUDA_DISABLE_PTX_JIT=1 \
  PATH="$cuda_root/bin:$PATH" \
  LD_LIBRARY_PATH="$candidate:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
  GENERATIVEQC_LIBRARY="$candidate/libgenerativeqc.so.0.1.0" \
  PYTHONPATH="$CANDIDATE_CHECKOUT/python:$CANDIDATE_CHECKOUT" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$REFERENCE_PYTHON" "$driver_directory/compare-native-ab.py" \
  --baseline "$baseline" --candidate "$candidate" \
  --baseline-source f37f96a1c3fee0da04ea0cc2c490f04140d65ae6 \
  --candidate-source 5aa703721fdce6a220838676af615d574f0fed05 \
  --reference-python "$REFERENCE_PYTHON" --repeats 5 --output "$output"

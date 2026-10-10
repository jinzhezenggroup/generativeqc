#!/usr/bin/env bash
# Standalone diagnostic qualification; no CMake/runtime/default modifications.
set -euo pipefail
qualification_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qualification_build=${1:?usage: qualify_ks_matrix_provider.sh ABSOLUTE_NEW_OUTPUT CUDA_ROOT SM_ARCH PYTHON}
qualification_cuda=${2:?explicit CUDA toolkit root required}
qualification_arch=${3:?explicit target such as 90 required}
qualification_python=${4:?explicit Python interpreter required}
qualification_mode=${5:-run}
[[ $qualification_mode = run || $qualification_mode = build-only ]]
[[ $qualification_build = /* && $qualification_cuda = /* && $qualification_python = /* ]]
[[ $qualification_arch =~ ^[0-9]+$ ]]
[[ ! -e $qualification_build ]]
mkdir -p -- "$qualification_build"
qualification_nvcc="$qualification_cuda/bin/nvcc"
qualification_cxx=$(command -v g++)
if command -v sccache >/dev/null; then
  qualification_cache=$(command -v sccache)
elif command -v ccache >/dev/null; then
  qualification_cache=$(command -v ccache)
else
  printf 'A verified compiler cache is required.\n' >&2
  exit 2
fi
"$qualification_cache" --version > "$qualification_build/cache-version.txt"
"$qualification_cache" --show-stats > "$qualification_build/cache-before.txt"
"$qualification_nvcc" --version > "$qualification_build/nvcc-version.txt"
"$qualification_cxx" --version > "$qualification_build/cxx-version.txt"
export CCACHE_BASEDIR="$qualification_root"
qualification_includes=(-I"$qualification_root/src" -I"$qualification_root/include" -I"$qualification_cuda/include")
for qualification_unit in matrix_library runtime_support; do
  "$qualification_cache" "$qualification_cxx" -std=c++20 -O3 -fPIC -MD \
    -MF "$qualification_build/$qualification_unit.d" "${qualification_includes[@]}" \
    -c "$qualification_root/src/scf/cuda/$qualification_unit.cpp" -o "$qualification_build/$qualification_unit.o"
done
"$qualification_cache" "$qualification_nvcc" -std=c++20 -O3 -arch="sm_$qualification_arch" \
  -ccbin "$qualification_cxx" -MD -MF "$qualification_build/scf_matrix_kernels.d" \
  "${qualification_includes[@]}" -c "$qualification_root/src/scf/cuda/scf_matrix_kernels.cu" \
  -o "$qualification_build/scf_matrix_kernels.o"
"$qualification_cache" "$qualification_nvcc" -std=c++20 -O3 -arch="sm_$qualification_arch" \
  -ccbin "$qualification_cxx" -MD -MF "$qualification_build/driver.d" \
  "${qualification_includes[@]}" -c "$qualification_root/tests/native/test_ks_matrix_provider_cuda.cu" \
  -o "$qualification_build/driver.o"
"$qualification_nvcc" -ccbin "$qualification_cxx" "$qualification_build/driver.o" \
  "$qualification_build/matrix_library.o" "$qualification_build/runtime_support.o" \
  "$qualification_build/scf_matrix_kernels.o" -lcublas \
  -Xlinker=--wrap=cublasCreate_v2 -Xlinker=--wrap=cudaMemGetInfo \
  -o "$qualification_build/matrix-qualification"
"$qualification_cache" --show-stats > "$qualification_build/cache-after.txt"
ldd "$qualification_build/matrix-qualification" > "$qualification_build/ldd.txt"
git -C "$qualification_root" rev-parse HEAD > "$qualification_build/source-head.txt"
git -C "$qualification_root" ls-files --stage -- src include cmake CMakeLists.txt \
  benchmarks/ks_matrix_provider_qualification.py tests/native/test_ks_matrix_provider_cuda.cu \
  tools/qualify_ks_matrix_provider.sh > "$qualification_build/source-blobs-modes.txt"
git -C "$qualification_root" status --porcelain --untracked-files=no > "$qualification_build/source-status.txt"
if [[ $qualification_mode = build-only ]]; then
  exit 0
fi
nvidia-smi -q > "$qualification_build/nvidia-smi.txt"
qualification_status=0
"$qualification_python" "$qualification_root/benchmarks/ks_matrix_provider_qualification.py" \
  run --output "$qualification_build/run" --driver "$qualification_build/matrix-qualification" || qualification_status=$?
"$qualification_python" "$qualification_root/benchmarks/ks_matrix_provider_qualification.py" \
  identity --output "$qualification_build" --compiler "$qualification_cxx" \
  --cuda "$qualification_cuda" --cache "$qualification_cache"
exit "$qualification_status"

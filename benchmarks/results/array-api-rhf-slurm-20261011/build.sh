#!/usr/bin/env bash
set -euo pipefail
root=/home/jzzeng/codes/qc-array-api-rhf-d15383b49-20261011
cd "$root"
evidence="${GENERATIVEQC_BUILD_EVIDENCE:-$root/.artifacts/qualification}"
mkdir -p "$evidence"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/codes/wb97m-20261002/ccache/usr/bin:/usr/bin:/bin
export NVCC_CCBIN=/usr/bin/g++-11
export CCACHE_BASEDIR="$root"
export PYTHONPATH="$root/python:$root"
printf 'job=%s node=%s visible=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "${CUDA_VISIBLE_DEVICES-unset}" | tee "$evidence/build-scheduler.txt"
ccache --version | tee "$evidence/cache-version.txt"
ccache --show-stats > "$evidence/cache-before-build.txt"
nvcc --version > "$evidence/nvcc-version.txt"
cmake --version > "$evidence/cmake-version.txt"
python -c 'import sys,numpy,pytest; print(sys.version); print("numpy", numpy.__version__, "pytest", pytest.__version__)' > "$evidence/build-python-version.txt"
cmake -S . -B build/cuda-release-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/jzzeng/miniconda3/bin/c++ \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DGENERATIVEQC_COMPILER_CACHE="$(command -v ccache)" \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=8 \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=8 \
  -DGENERATIVEQC_AOT_PROFILE=sm_120 \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=OFF \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF \
  -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF \
  -DGENERATIVEQC_BUILD_TESTS=OFF \
  -DGENERATIVEQC_BUILD_CLI=OFF \
  -DPython_EXECUTABLE=/home/jzzeng/miniconda3/bin/python \
  2>&1 | tee "$evidence/configure.log"
ninja -C build/cuda-release-sm120 -t commands > "$evidence/compiler-commands.txt"
printf 'cache-backed compiler commands: '
grep -c 'ccache .* -c ' "$evidence/compiler-commands.txt"
timeout 1500 cmake --build build/cuda-release-sm120 --parallel 16 \
  2>&1 | tee "$evidence/build.log"
ccache --show-stats > "$evidence/cache-after-build.txt"
sha256sum build/cuda-release-sm120/libgenerativeqc.so > "$evidence/library-sha256.txt"
ldd build/cuda-release-sm120/libgenerativeqc.so > "$evidence/library-ldd.txt"
cp build/cuda-release-sm120/CMakeCache.txt "$evidence/CMakeCache.txt"
printf 'BUILD COMPLETE\n'

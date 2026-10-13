#!/usr/bin/env bash
# One immutable native/AOT build shared by the incremental and crossover studies.
set -euo pipefail
closeout_root=/data/jzzeng/qc-closeout-20261013
closeout_python=/data/jzzeng/qc-932-validation-env-20261011/bin/python
closeout_tools=/data/jzzeng/qc-provider-1886-tools/bin
closeout_cmake=/data/jzzeng/qc-932-validation-env-20261011/bin/cmake
cd "$closeout_root"
test "$(git rev-parse HEAD)" = e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c
test -z "$(git status --porcelain)"
mkdir -p .artifacts/closeout-build
export PATH="/group/software/cuda-12.9.1/bin:$closeout_tools:$PATH"
export PYTHONPATH="$closeout_root/python:$closeout_root"
export CCACHE_BASEDIR="$closeout_root"
ccache --version > .artifacts/closeout-build/cache-version.txt
ccache --show-stats > .artifacts/closeout-build/cache-before.txt
"$closeout_cmake" -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_BUILD_TESTS=OFF \
  -DGENERATIVEQC_BUILD_CLI=OFF \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120 \
  -DGENERATIVEQC_AOT_PROFILE=portable \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  -DGENERATIVEQC_STATIONARY_AOT_PROFILES=pbe0_rks \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$closeout_tools/ccache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$closeout_tools/ccache" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DPython3_EXECUTABLE="$closeout_python" \
  -DPython_EXECUTABLE="$closeout_python" \
  > .artifacts/closeout-build/configure.log 2>&1
ninja -C build-cuda -t commands generativeqc > .artifacts/closeout-build/compile-commands.txt
"$closeout_cmake" --build build-cuda --parallel 8 > .artifacts/closeout-build/build.log 2>&1
"$closeout_cmake" --install build-cuda --prefix "$closeout_root/.artifacts/install" \
  > .artifacts/closeout-build/install.log 2>&1
sha256sum .artifacts/install/lib/libgenerativeqc*.so* \
  > .artifacts/closeout-build/binaries.sha256
ccache --show-stats > .artifacts/closeout-build/cache-after.txt
printf 'BUILD_COMPLETE\n'

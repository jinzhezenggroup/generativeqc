#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$root/source"
export CCACHE_BASEDIR="$PWD" PYTHONPATH="$PWD/python:$PWD:$deps"
out="$root/build-auto"
mkdir -p "$out"
ccache --version > "$out/ccache-version.txt"
ccache --show-stats > "$out/ccache-before.txt"
trap 'ccache --show-stats > "$out/ccache-after.txt"' EXIT
cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DPython3_EXECUTABLE=/home/jzzeng/miniconda3/bin/python \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
  -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  > "$out/configure.log" 2>&1
grep -m4 'LAUNCHER = ccache' build-cuda/build.ninja > "$out/compiler-launchers.txt"
cmake --build build-cuda --target generativeqc -j12 > "$out/build.log" 2>&1
ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
  -I/group/software/cuda-12.9.1/include \
  -c benchmarks/df_ccsdt_force_endpoint.cpp -o "$out/endpoint.o"
g++ "$out/endpoint.o" build-cuda/libgenerativeqc.so \
  -Wl,-rpath,"$PWD/build-cuda" -o "$out/endpoint"
sha256sum build-cuda/libgenerativeqc.so "$out/endpoint" \
  benchmarks/df_ccsdt_force_endpoint.cpp > "$out/binaries.sha256"
{ date --iso-8601=seconds; hostname; g++ --version | head -1; nvcc --version;
  cmake --version | head -1; ccache --version; } > "$out/toolchain.txt"
printf '%s\n' complete > "$out/status.txt"

#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/issue2516-20261027
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$root/source"
export CCACHE_BASEDIR="$PWD" PYTHONPATH="$PWD/python:$PWD:$deps"
ccache --version > "$root/evidence/ccache-version.txt"
ccache --show-stats > "$root/evidence/ccache-before.txt"
trap 'ccache --show-stats > "$root/evidence/ccache-after.txt"' EXIT
cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DPython3_EXECUTABLE=/home/jzzeng/miniconda3/bin/python \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
  -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  > "$root/evidence/configure.log" 2>&1
cmake --build build-cuda --target generativeqc -j12 > "$root/evidence/build.log" 2>&1
ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
  -I/group/software/cuda-12.9.1/include \
  -c benchmarks/df_ccsdt_force_endpoint.cpp -o "$root/evidence/endpoint.o"
g++ "$root/evidence/endpoint.o" build-cuda/libgenerativeqc.so \
  -Wl,--enable-new-dtags,-rpath,"$PWD/build-cuda" -o "$root/evidence/endpoint"
sha256sum build-cuda/libgenerativeqc.so "$root/evidence/endpoint" \
  benchmarks/df_ccsdt_force_endpoint.cpp > "$root/evidence/binaries.sha256"

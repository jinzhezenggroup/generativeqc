#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_ROOT:?Point at the frozen baseline or candidate source checkout}"
cache=$(command -v sccache || command -v ccache)
"$cache" --version
"$cache" --show-stats
export CCACHE_BASEDIR="$SOURCE_ROOT"
cmake -S "$SOURCE_ROOT" -B "$SOURCE_ROOT/build/cuda-release-sm120" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$cache" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache" \
    -DGENERATIVEQC_ENABLE_CUDA=ON \
    -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
    -DGENERATIVEQC_AOT_PROFILE=sm_120 \
    -DGENERATIVEQC_BUILD_TESTS=OFF \
    -DGENERATIVEQC_BUILD_CLI=OFF \
    -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
    -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
    -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF \
    '-DGENERATIVEQC_STATIONARY_AOT_PROFILES=r2scan_rks;pbe0_rks'
cmake --build "$SOURCE_ROOT/build/cuda-release-sm120" --parallel 8
"$cache" --show-stats

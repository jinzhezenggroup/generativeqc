#!/usr/bin/env bash
set -euo pipefail
root=${1:?Provide an enclosing fixture root containing the measured source checkout}
bundle=$(cd "$(dirname "$0")" && pwd)
cuda=${CUDA_HOME:-/group/software/cuda-12.9.1}
export PATH="$cuda/bin:$PATH"
export CCACHE_BASEDIR="$root/source"
mkdir -p "$root/bin"
cd "$root/source"
ccache --version > "$root/ccache-version.txt"
ccache --show-stats > "$root/ccache-before.txt"
trap 'ccache --show-stats > "$root/ccache-after.txt"' EXIT
cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF -DGENERATIVEQC_BUILD_TESTS=OFF \
  -DGENERATIVEQC_BUILD_CLI=OFF -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  -DGENERATIVEQC_AOT_PROFILE=portable_cuda -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  > "$root/configure.log" 2>&1
cmake --build build-cuda --target generativeqc \
  generativeqc_stationary_pbe0_rks_spd_manifest -j"${BUILD_JOBS:-12}" \
  > "$root/build.log" 2>&1
ccache g++ -std=c++20 -O2 -Isrc -Iinclude -I"$cuda/include" \
  -c tests/native/ks_phase_values_probe.cpp -o "$root/bin/probe.o"
g++ "$root/bin/probe.o" -Lbuild-cuda -lgenerativeqc \
  -Wl,-rpath,'$ORIGIN' -L"$cuda/lib64" -lcudart -o "$root/bin/ks_phase_values_probe"
cp build-cuda/libgenerativeqc.so build-cuda/libgenerativeqc_stationary_pbe0_rks_spd.so \
  build-cuda/generativeqc_stationary_pbe0_rks_spd.json "$root/bin/"
ln -s libgenerativeqc.so "$root/bin/libgenerativeqc.so.0"
cp "$bundle/observe-default.py" "$bundle/run-default.sh" "$root/"
gzip -dc "$bundle/reference.json.gz" > "$root/reference.json"
sha256sum "$root/bin/"*so* "$root/bin/ks_phase_values_probe" \
  "$root/bin/generativeqc_stationary_pbe0_rks_spd.json" > "$root/binary-SHA256SUMS"
printf '%s\n' complete > "$root/build-status.txt"

#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/readme-defaults-20261020
export PATH=/home/jzzeng/.local/bin:/data/jzzeng/qc-provider-1886-tools/bin:/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export CCACHE_BASEDIR="$root/source"
launcher=/home/jzzeng/.local/bin/sccache
"$launcher" --version > "$root/receipts/cache-version.txt"
"$launcher" --show-stats > "$root/receipts/cache-final-before.txt"
cmake -S "$root/source" -B "$root/source/build/cuda-release-sm120" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
 -DCMAKE_CXX_COMPILER_LAUNCHER="$launcher" -DCMAKE_CUDA_COMPILER_LAUNCHER="$launcher" \
 -DGENERATIVEQC_BUILD_TESTS=ON -DGENERATIVEQC_BUILD_CLI=ON -Dnlohmann_json_DIR="$root/deps" -DCMAKE_PREFIX_PATH=/home/jzzeng/miniconda3 \
 -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_AOT_PROFILE=sm_120 \
 -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
 -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
 -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
 -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
 -DGENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=OFF \
 -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON \
 -DGENERATIVEQC_HESSIAN_AOT_INVENTORY="$root/source/manifests/hessian_aot_development.json" \
 -DGENERATIVEQC_AOT_COMPILE_JOBS=4 -DGENERATIVEQC_CUDA_COMPILE_JOBS=8 \
 -DPython_EXECUTABLE=/home/jzzeng/codes/wb97m-20261002/gpu-env/bin/python
cmake --build "$root/source/build/cuda-release-sm120" --target generativeqc_cli generativeqc_stationary_pbe_rks_manifest generativeqc_stationary_pbe0_rks_manifest generativeqc_stationary_r2scan_rks_manifest -j 16
"$launcher" --show-stats > "$root/receipts/cache-final-after.txt"
ninja -C "$root/source/build/cuda-release-sm120" -t commands > "$root/receipts/compiler-commands.txt"
sha256sum "$root/source/build/cuda-release-sm120/libgenerativeqc.so" > "$root/receipts/library.sha256"

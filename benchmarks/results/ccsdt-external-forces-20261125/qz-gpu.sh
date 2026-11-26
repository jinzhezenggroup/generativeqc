#!/usr/bin/env bash
# Build and measure the public CUDA endpoints on one allocated qz H100.
set -euo pipefail
root=$PWD
out="$root/$1"
base=$(dirname "$(dirname "$root")")
env_root="$base/h100-portability/20261027-fd06"
python="$env_root/env/bin/python"
sccache="$env_root/env/bin/sccache"
cuda=/opt/generativeqc-reuse/cuda-12.9
provider="$env_root/env/lib/python3.11/site-packages/scipy_openblas32/lib"
source="$root/source"
build="$out/build"
export GENERATIVEQC_LIBRARY="$build/libgenerativeqc.so"
export LD_LIBRARY_PATH="$build:$provider:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export PATH="$env_root/env/bin:$cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$source/python:$source"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SCCACHE_DIR="$env_root/cache/sccache"
export GENERATIVEQC_STATIONARY_CACHE="$out/runtime-cache"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$out/jit-owners"
export NVCC_CCBIN=/usr/bin/g++-12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$out"
exec > "$out/driver.log" 2>&1
echo "START $(date -u +%FT%TZ)"
trap 'status=$?; echo "EXIT $status $(date -u +%FT%TZ)"; exit "$status"' EXIT
(
    set -euo pipefail
    tmp=$(mktemp -d /tmp/gqc-3167-cache.XXXXXXXX)
    export SCCACHE_SERVER_UDS="$tmp/socket" SCCACHE_IDLE_TIMEOUT=1200
    "$sccache" --version
    SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 "$sccache" > "$out/cache-server.log" 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; rm -f "$SCCACHE_SERVER_UDS"; rmdir "$tmp"' EXIT
    for count in {1..100}; do
        if [[ -S "$SCCACHE_SERVER_UDS" ]]; then break; fi
        kill -0 "$server"
        sleep .1
    done
    test -S "$SCCACHE_SERVER_UDS"
    "$sccache" --show-stats > "$out/cache-before.txt"
    "$python" -m cmake -S "$source" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
        -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
        -DCMAKE_CXX_COMPILER_LAUNCHER="$sccache" \
        -DCMAKE_CUDA_COMPILER_LAUNCHER="$sccache" \
        -DPython3_EXECUTABLE="$python" -DOpenBLAS_DIR="$provider/cmake/openblas" \
        -DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas \
        -DCMAKE_CUDA_ARCHITECTURES=90 -DGENERATIVEQC_CUDA_ARCHITECTURES=90 \
        -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
        -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
        -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF \
        -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
        -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF \
        -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
        -DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=ON \
        -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 -DGENERATIVEQC_AOT_COMPILE_JOBS=4 \
        -DGENERATIVEQC_AOT_SHARDS=8 > "$out/configure.log" 2>&1
    "$python" -m ninja -C "$build" -t commands generativeqc > "$out/compiler-commands.txt"
    "$python" - "$out/compiler-commands.txt" "$sccache" <<'PY'
from pathlib import Path
import sys
commands = Path(sys.argv[1]).read_text().splitlines()
compiles = [line for line in commands if " -c " in line and ("g++-12" in line or "/nvcc" in line)]
if not compiles or any(sys.argv[2] not in line for line in compiles):
    raise RuntimeError("An actual C++/CUDA compile is missing the verified sccache launcher")
print("Verified compiler launcher on", len(compiles), "commands")
PY
    "$python" -m ninja -C "$build" -j 8 generativeqc > "$out/build.log" 2>&1
    "$sccache" --show-stats > "$out/cache-after.txt"
)
sha256sum "$GENERATIVEQC_LIBRARY" > "$out/library-sha256.txt"
"$cuda/bin/nvcc" --version > "$out/cuda-version.txt"
echo "BUILD_COMPLETE $(date -u +%FT%TZ)"
for arm in native-df native-conv; do
    "$python" -B -u run.py --source "$source" \
        --arm "$arm" --output "$out/$arm" --threads 1 --timeout 1200 \
        > "$out/$arm.log" 2>&1
done

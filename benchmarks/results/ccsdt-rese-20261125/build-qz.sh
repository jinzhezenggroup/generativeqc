#!/usr/bin/env bash
# Build the native core and CLI together from the pinned master source.
set -euo pipefail
root=$(pwd -P)
source_dir=$(realpath "$1")
environment=$(realpath "$2")
deps=$(realpath "$3")
cuda=/opt/generativeqc-reuse/cuda-12.9
python="$environment/env/bin/python"
launcher="$environment/env/bin/sccache"
openblas="$environment/env/lib/python3.11/site-packages/scipy_openblas32/lib"
export PATH="$environment/env/bin:$cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$deps/usr/lib/x86_64-linux-gnu:$openblas:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export SCCACHE_DIR="$environment/cache/sccache" SCCACHE_IDLE_TIMEOUT=1200
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
receipt="$root/build-receipts"
mkdir "$receipt"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > "$receipt/gpu.txt"
lscpu > "$receipt/cpu.txt"
cp "$deps/packages.sha256" "$receipt/cli-dependency-packages.sha256"
"$cuda/bin/nvcc" --version > "$receipt/nvcc.txt"
g++-12 --version > "$receipt/host-compiler.txt"
"$python" --version > "$receipt/python.txt"
"$launcher" --version | tee "$receipt/launcher.txt"
runtime=$(mktemp -d /tmp/gqc-3166.XXXXXXXX)
export SCCACHE_SERVER_UDS="$runtime/sccache.sock"
SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 "$launcher" > "$receipt/cache-server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" || true; rm -rf "$runtime"' EXIT
for attempt in $(seq 1 100); do
  if test -S "$SCCACHE_SERVER_UDS"; then break; fi
  kill -0 "$server"
  sleep 0.1
done
"$launcher" --show-stats > "$receipt/cache-before.txt"
# Use the reviewed headers already installed in the frozen environment.
json_include=$("$python" -c 'import nlohmann_json; print(nlohmann_json.get_include())')
mkdir "$runtime/json"
cat > "$runtime/json/nlohmann_jsonConfig.cmake" <<CONFIG
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
set_property(TARGET nlohmann_json::nlohmann_json PROPERTY INTERFACE_INCLUDE_DIRECTORIES "$json_include")
CONFIG
cat > "$runtime/json/nlohmann_jsonConfigVersion.cmake" <<'CONFIG'
set(PACKAGE_VERSION "3.11.3")
if(PACKAGE_FIND_VERSION VERSION_LESS_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()
CONFIG
args=(
  -S "$source_dir" -B "$root/build" -G Ninja
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
  "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
  "-DCMAKE_CXX_COMPILER_LAUNCHER=$launcher" "-DCMAKE_CUDA_COMPILER_LAUNCHER=$launcher"
  "-DPython3_EXECUTABLE=$python" "-DOpenBLAS_DIR=$openblas/cmake/openblas"
  "-Dnlohmann_json_DIR=$runtime/json" -DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas
  "-DOPENSSL_INCLUDE_DIR=$deps/usr/include"
  "-DOPENSSL_CRYPTO_LIBRARY=$deps/usr/lib/x86_64-linux-gnu/libcrypto.so"
  "-DLibArchive_INCLUDE_DIR=$deps/usr/include"
  "-DLibArchive_LIBRARY=$deps/usr/lib/x86_64-linux-gnu/libarchive.so"
  "-DCMAKE_CXX_FLAGS=-isystem $deps/usr/include/x86_64-linux-gnu"
  -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=OFF
  -DCMAKE_CUDA_ARCHITECTURES=90 -DGENERATIVEQC_CUDA_ARCHITECTURES=90
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON
  -DGENERATIVEQC_AOT_PROFILE=portable_cuda
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
  -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=OFF
  -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF
  -DGENERATIVEQC_BUILD_CLI=ON -DGENERATIVEQC_BUILD_TESTS=OFF
  -DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=ON -DGENERATIVEQC_CUDA_COMPILE_JOBS=6
  -DGENERATIVEQC_AOT_COMPILE_JOBS=6 -DGENERATIVEQC_AOT_SHARDS=8
)
printf '%q ' "$python" -m cmake "${args[@]}" > "$receipt/configure-command.txt"
printf '\n' >> "$receipt/configure-command.txt"
"$python" -m cmake "${args[@]}" > "$receipt/configure.log" 2>&1
"$python" -m ninja -C build -t commands generativeqc_cli > "$receipt/compiler-commands.txt"
awk -v launcher="$launcher" '
    / -c / && (/g\+\+-12/ || /\/nvcc/) {seen++; if (!index($0, launcher)) bad++}
    END {exit (!seen || bad)}
' "$receipt/compiler-commands.txt"
set +e
"$python" -m ninja -C build -j18 generativeqc_cli > "$receipt/build.log" 2>&1
status=$?
set -e
"$launcher" --show-stats > "$receipt/cache-after.txt"
test "$status" -eq 0
ldd build/generativeqc > "$receipt/linkage.txt"
if grep -q '=> not found' "$receipt/linkage.txt"; then exit 1; fi
"$python" - "$source_dir" "$root" <<'PY' > "$receipt/source-identity.txt"
import ctypes
import importlib.util
from pathlib import Path
import sys

source, root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("identity", source / "tools/generate_build_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
inventory = identity._inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
expected = identity._source_identity(source, inventory)
lib = ctypes.CDLL(str(root / "build/libgenerativeqc.so"))
lib.generativeqc_get_source_identity.restype = ctypes.c_char_p
actual = lib.generativeqc_get_source_identity().decode()
print(f"staged: {expected}\nloaded: {actual}")
if actual != expected:
    raise ValueError("native source identity mismatch")
PY
sha256sum build/generativeqc build/libgenerativeqc.so > "$receipt/binaries.sha256"

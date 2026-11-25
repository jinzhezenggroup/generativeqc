#!/usr/bin/env bash
# Build the pinned source in the verified qz H100 image.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
assets=$(dirname -- "$root")
runtime="$assets/h100-portability/20261027-fd06"
bin="$runtime/env/bin"
provider="$runtime/env/lib/python3.11/site-packages/scipy_openblas32/lib"
cuda=/opt/generativeqc-reuse/cuda-12.9
out="$root/build"
receipt="$root/build-receipts"
mkdir "$receipt"
# Upload and extraction go through the shared CPU notebook. The GPU job can
# be scheduled before transfer completion, so wait for the verified handoff.
for (( i=0; i<120; i++ )); do
    [[ ! -f $root/source-staged.sha256 ]] || break
    sleep 5
done
(cd "$root" && sha256sum -c source-staged.sha256)
test -f "$root/source/manifests/public_methods.json"
export PATH="$bin:$cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$provider:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVCC_CCBIN=/usr/bin/g++-12
export SCCACHE_DIR="$runtime/cache/sccache" SCCACHE_IDLE_TIMEOUT=3600
socket_dir=$(mktemp -d /tmp/gqc3168-cache.XXXXXX)
export SCCACHE_SERVER_UDS="$socket_dir/server.sock"
cache_pid=
finish() {
    status=$?
    printf '%s\n' "$status" > "$receipt/exit-code.txt"
    if [[ -n $cache_pid ]]; then
        "$bin/sccache" --stop-server > "$receipt/cache-stop.txt" 2>&1 || true
        kill "$cache_pid" 2>/dev/null || true
        wait "$cache_pid" 2>/dev/null || true
    fi
    rm -f -- "$SCCACHE_SERVER_UDS"
    rmdir -- "$socket_dir"
    trap - EXIT
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$bin/sccache" --version > "$receipt/cache-version.txt"
grep -Fx 'sccache 0.16.0' "$receipt/cache-version.txt"
"$cuda/bin/nvcc" --version > "$receipt/nvcc-version.txt"
/usr/bin/g++-12 --version > "$receipt/gxx-version.txt"
SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 "$bin/sccache" > "$receipt/cache-server.txt" 2>&1 &
cache_pid=$!
for (( i=0; i<100; i++ )); do
    [[ ! -S $SCCACHE_SERVER_UDS ]] || break
    kill -0 "$cache_pid"
    sleep 0.1
done
test -S "$SCCACHE_SERVER_UDS"
"$bin/sccache" --show-stats > "$receipt/cache-before.txt"
cmake=(
    "$bin/python" -m cmake -S "$root/source" -B "$out" -G Ninja
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
    "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
    "-DCMAKE_CXX_COMPILER_LAUNCHER=$bin/sccache" "-DCMAKE_CUDA_COMPILER_LAUNCHER=$bin/sccache"
    "-DPython3_EXECUTABLE=$bin/python" "-DOpenBLAS_DIR=$provider/cmake/openblas"
    -DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas
    -DCMAKE_CUDA_ARCHITECTURES=90 -DGENERATIVEQC_CUDA_ARCHITECTURES=90
    -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_CUDA_FAST_COMPILE=ON
    -DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
    -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=OFF -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF
    -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF
    -DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=ON -DGENERATIVEQC_CUDA_COMPILE_JOBS=4
)
printf '%q ' "${cmake[@]}" > "$receipt/configure-command.txt"
printf '\n' >> "$receipt/configure-command.txt"
date -u +%FT%TZ > "$receipt/started-utc.txt"
"${cmake[@]}" > "$receipt/configure.txt" 2>&1
"$bin/python" -m ninja -C "$out" -t commands generativeqc > "$receipt/compiler-commands.txt"
"$bin/python" - "$receipt/compiler-commands.txt" "$bin/sccache" <<'PY'
import sys
from pathlib import Path
lines = [
    line for line in Path(sys.argv[1]).read_text().splitlines()
    if " -c " in line and ("/g++-12" in line or "/nvcc" in line)
]
assert lines and all(sys.argv[2] in line for line in lines)
print(f"{len(lines)} host/CUDA compiler commands use sccache")
PY
"$bin/python" -m ninja -C "$out" -j 16 generativeqc > "$receipt/build.txt" 2>&1
"$bin/sccache" --show-stats > "$receipt/cache-after.txt"
LD_LIBRARY_PATH="$out:$LD_LIBRARY_PATH" "$bin/python" - "$root" <<'PY'
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root = Path(sys.argv[1])
source = root / "source"
sys.path.insert(0, str(source / "tools"))
from generate_build_identity import _inventory, _source_identity

revision = "22f72d0692a760c2892cd7791022584d9ae1c3e5"
archive_path = root / "source.tar.gz"
with tarfile.open(archive_path) as archive:
    assert archive.pax_headers["comment"] == revision

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

source_id = _source_identity(
    source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
)
library_path = root / "build/libgenerativeqc.so"
library = ctypes.CDLL(str(library_path))
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == source_id
receipt = {
    "source_revision": revision,
    "native_source_identity": source_id,
    "source_archive_sha256": sha(archive_path),
    "library_sha256": sha(library_path),
    "build_driver_sha256": sha(root / "build_qz.sh"),
    "cmake_cache_sha256": sha(root / "build/CMakeCache.txt"),
}
(root / "build-receipts/identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
date -u +%FT%TZ > "$receipt/finished-utc.txt"

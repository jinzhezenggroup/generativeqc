#!/usr/bin/env bash
set -euo pipefail
trial=/data/jzzeng/direct-gap-master-20261105
source=/data/jzzeng/issue2987-defaults-20261105/source
out="$trial/candidate-build"
exec 9>/data/jzzeng/direct-range-anchor-20261105/compile-owner.lock
flock -n 9
mkdir "$out" "$trial/master-source"
export PATH=/home/jzzeng/.local/bin:/home/jzzeng/miniconda3/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cache=/home/jzzeng/.local/bin/sccache
"$cache" --version > "$out/cache-version.txt"
export SCCACHE_SERVER_UDS="$trial/tmp/candidate-cache-$$.sock" SCCACHE_IDLE_TIMEOUT=1800
setsid env SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 "$cache" > "$out/cache-server.log" 2>&1 &
server=$!
finish() {
  status=$?
  trap - EXIT
  "$cache" --show-stats > "$out/cache-after.txt" 2>&1 || true
  kill -- "-$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  printf '%s\n' "$status" > "$out/exit-code.txt"
  date -Is > "$out/completed-at.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for attempt in {1..100}; do
  [[ -S "$SCCACHE_SERVER_UDS" ]] && break
  kill -0 "$server"
  sleep .1
done
[[ -S "$SCCACHE_SERVER_UDS" ]]
"$cache" --show-stats > "$out/cache-before.txt"
date -Is > "$out/started-at.txt"
tar -C "$source" --exclude=./build --exclude=./.artifacts --exclude=./.git -cf - . | tar -C "$trial/master-source" -xf -
ln -s "$trial/master-image" "$trial/master-source/build"
tar -xzf "$trial/candidate-overlay.tar.gz" -C "$source"
cd "$source"
python - "$trial/candidate-source-identity.txt" <<'PY'
from pathlib import Path
import sys
from tools.generate_build_identity import _inventory, _source_identity
root = Path.cwd()
identity = _source_identity(root, _inventory(root, root / 'cmake/GenerativeQCSourceIdentity.json'))
assert identity == Path(sys.argv[1]).read_text().strip(), identity
print('Authenticated isolated compact-MD source', identity, flush=True)
PY
timeout --kill-after=30s 1800 cmake --build build --verbose --target generativeqc -j 12 > "$out/build.log" 2>&1
export LD_LIBRARY_PATH="$source/build:/data/jzzeng/portable-integration-20261101/provider/scipy_openblas32/lib:/group/software/cuda-12.9.1/lib64"
python - "$trial/candidate-source-identity.txt" "$out/identity.json" <<'PY'
import ctypes, hashlib, json, sys
from pathlib import Path
root = Path.cwd()
identity = Path(sys.argv[1]).read_text().strip()
library = ctypes.CDLL(str(root / 'build/libgenerativeqc.so'))
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == identity
binary = root / 'build/libgenerativeqc.so'
Path(sys.argv[2]).write_text(json.dumps({
    'source_identity': identity,
    'base_commit': 'a970014e8de544aa54c6e0662eda32ebc5e5388f',
    'library_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
}, indent=2) + '\n')
PY
tar -czf "$trial/candidate-runtime.tar.gz" -C "$source" build/libgenerativeqc.so build/libgenerativeqc.so.0 build/libgenerativeqc.so.0.1.0
sha256sum "$trial/candidate-runtime.tar.gz" > "$out/runtime.sha256"
touch "$trial/candidate-build-ready"

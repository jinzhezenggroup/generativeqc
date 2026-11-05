#!/usr/bin/env bash
set -euo pipefail
: "${LSB_JOBID:?actual finite LSF GPU allocation required}"
: "${CUDA_VISIBLE_DEVICES:?scheduler visibility required}"
assigned=$CUDA_VISIBLE_DEVICES
base=/tmp/generativeqc-ecnu-a100-tzhu-e-20261029
shared="$base/issue2987-hold-fix-20261105"
trial="$base/direct-gap-master-20261105"
source="$trial/source"
out="$trial/results-$LSB_JOBID"
transfer=http://login02:18789
mkdir -p "$trial/python-runtime" "$source"
mkdir "$out"
exec > "$out/driver.log" 2>&1
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$out/exit-code.txt"
  date -Is > "$out/completed-at.txt"
  tar -czf "$out.tar.gz" -C "$out" .
  curl --fail --max-time 90 --data-binary @"$out.tar.gz" "$transfer/results-$LSB_JOBID.tar.gz" || status=74
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is > "$out/started-at.txt"
hostname > "$out/hostname.txt"
cp "$0" "$out/runner.sh"
for name in master-overlay.tar.gz master-runtime.tar.gz master-source-identity.txt endpoint.py master-image-identity.json; do
  if [[ ! -s "$trial/$name" ]]; then
    curl --fail --max-time 180 "$transfer/$name" -o "$trial/$name"
  fi
done
if [[ ! -s "$source/build/libgenerativeqc.so" ]]; then
  tar -C "$base/direct-range-default-20261105/source" --exclude=./build --exclude=./.artifacts -cf - . | tar -C "$source" -xf -
  tar -xzf "$trial/master-overlay.tar.gz" -C "$source"
  tar -xzf "$trial/master-runtime.tar.gz" -C "$source"
fi
source "$base/native-env.sh"
frozen="$base/cmake-python-openblas-20261101"
runtime_ld="$shared/glibc:$shared/deps:$source/build:$shared/source/scipy_openblas32/lib:$base/env/lib:$base/env/targets/x86_64-linux/lib"
loader="$shared/glibc/ld-linux-x86-64.so.2"
cp "$base/env/bin/python" "$trial/python-runtime/python"
"$base/env/bin/patchelf" --set-interpreter "$loader" --force-rpath --set-rpath "$runtime_ld" "$trial/python-runtime/python"
export PYTHONHOME="$base/env"
python=("$loader" --library-path "$runtime_ld" "$trial/python-runtime/python" -u -X faulthandler)
export PYTHONPATH="$source/python:$source:$frozen/provider:$base/results/stationary-parity/reference-packages"
export GENERATIVEQC_LIBRARY="$source/build/libgenerativeqc.so"
export CUDA_HOME="$base/env/targets/x86_64-linux" CUDA_PATH="$base/env/targets/x86_64-linux"
export CXX="$base/env/bin/x86_64-conda-linux-gnu-g++" NVCC_CCBIN="$CXX" CCACHE="$base/env/bin/sccache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GENERATIVEQC_PROFILE=off GENERATIVEQC_PROFILE_CACHE="$out/profile-cache"
export XDG_CACHE_HOME="$base/native-cache" CUPY_CACHE_DIR="$base/native-cupy"
unset GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES
unset GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_PAIR_SEPARABLE GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES
cd "$source"
"${python[@]}" - "$trial/master-source-identity.txt" "$trial/master-image-identity.json" "$out/identity.json" <<'PY'
import ctypes, hashlib, json, os, sys
from pathlib import Path
from tools.generate_build_identity import _inventory, _source_identity
root = Path.cwd()
identity = _source_identity(root, _inventory(root, root / 'cmake/GenerativeQCSourceIdentity.json'))
expected = json.loads(Path(sys.argv[2]).read_text())
assert identity == Path(sys.argv[1]).read_text().strip() == expected['source_identity']
library = ctypes.CDLL(os.environ['GENERATIVEQC_LIBRARY'])
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == identity
digest = hashlib.file_digest(open(os.environ['GENERATIVEQC_LIBRARY'], 'rb'), 'sha256').hexdigest()
assert digest == expected['library_sha256']
expected['scheduler'] = {name: os.environ.get(name) for name in ('LSB_JOBID', 'LSB_QUEUE', 'LSB_HOSTS', 'CUDA_VISIBLE_DEVICES')}
Path(sys.argv[3]).write_text(json.dumps(expected, indent=2) + '\n')
PY
failed=0
for arm in native reference; do
  status=0
  extra=(--method r2scan-3c)
  if [[ "$arm" = native ]]; then
    export GENERATIVEQC_DF_TRACE="$out/native-trace.jsonl"
    export GENERATIVEQC_DIRECT_FORCE_CENSUS="$out/native-force-census.jsonl"
  else
    unset GENERATIVEQC_DF_TRACE GENERATIVEQC_DIRECT_FORCE_CENSUS
    extra+=(--reference)
  fi
  timeout --kill-after=20s 480 "${python[@]}" "$trial/endpoint.py" "${extra[@]}" \
    --source "$(cat "$trial/master-source-identity.txt")" --inputs "$shared/source/inputs" \
    --output "$out/$arm.json" > "$out/$arm.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$out/$arm.exit-code.txt"
  if [[ "$status" != 0 ]]; then failed=1; fi
done
test "$CUDA_VISIBLE_DEVICES" = "$assigned"
exit "$failed"

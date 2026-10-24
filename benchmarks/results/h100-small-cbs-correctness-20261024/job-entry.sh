#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
R=/inspire/ssd/project/project-public/czxs25220150/h100-paper-environment-public-20261024-9dcc/runtime-v1
T=/inspire/ssd/project/project-public/czxs25220150/h100-small-cbs-correctness-20261024-e7f7/v2
P=/opt/generativeqc-reuse/python/bin/python3.11
read -r python_sha ignored_path < <(/usr/bin/sha256sum "$P")
[[ "$python_sha" == 1e761eb19d6f2594ab8dc64bd99ad4e1753589f3bf1ec199e4eef3aaa21e3930 ]]
"$P" -I -S -B - "$T/bootstrap-activation.json" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
script = Path('/opt/generativeqc-reuse/activate.sh')
data = script.read_bytes()
checksum = hashlib.sha256(data).hexdigest()
if len(data) != 398 or checksum != 'f788e97155d31786a79d719b33ecded54a7e87a39ed452720c12f88bcfba21ee' or stat.S_IMODE(script.stat().st_mode) != 0o644:
    raise RuntimeError('current activation bytes/mode differ; no source or SCI')
with Path(sys.argv[1]).open('x', encoding='utf-8') as stream:
    json.dump({'scope': 'new CPU-observed activation pin, not retrospective smoke provenance',
               'path': str(script), 'bytes': len(data), 'sha256': checksum,
               'mode': oct(stat.S_IMODE(script.stat().st_mode)), 'pid': os.getpid()}, stream)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
PY
source /opt/generativeqc-reuse/activate.sh
export PATH="$R/env/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_LIBRARY="$R/build/cbs-sm90-portable-attempt2/libgenerativeqc.so"
export GENERATIVEQC_JIT_SCCACHE_ROOT="$T/jit-slots"
export TMPDIR="$T/tmp" PIP_CACHE_DIR="$T/pip-cache"
export CUPY_CACHE_DIR="$T/cupy-cache" XDG_CACHE_HOME="$T/cache"
mkdir -p "$T/tmp" "$T/pip-cache" "$T/cupy-cache" "$T/cache"
cd "$T/scripts"
exec "$R/env/bin/python" -B "$T/scripts/supervisor.py" \
  --plan "$T/scripts/prepared-plan-v4.json" \
  --preflight "$T/preflight" --wait-release \
  --release "$T/root-release.json"

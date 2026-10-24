#!/usr/bin/env bash
# CPU-only byte/mode recovery and independent mathematical replay; no endpoints.
set -euo pipefail
root=/data/srxu/projects/gqc-runtime-cache-state-20261023
cd "$root"
: "${SLURM_JOB_ID:?requires real CPU Slurm allocation}"
export PATH="$root/runtime-env/bin:/usr/bin:/bin"
export PYTHONPATH="$root/source/python:$root/harness"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
tar -xzf linux-recovery/capsule.tar.gz -C linux-recovery
python "$root/harness/benchmarks/results/runtime-cache-state-20261023/reproduce.py" \
  --bundle "$root/linux-recovery/capsule" --output "$root/linux-recovery/replayed"
python - <<'PY'
import hashlib
import json
from pathlib import Path
root=Path('/data/srxu/projects/gqc-runtime-cache-state-20261023/linux-recovery/replayed/raw')
index=json.loads((root/'raw-index.json').read_bytes())
for row in index['files']:
    path=root/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    assert path.stat().st_size==row['bytes']
    assert f'{path.stat().st_mode & 0o777:04o}'==row['mode']
print(json.dumps({'original_files':len(index['files']), 'bytes_sha256_modes':'PASS', 'gpu_execution':'NOT_RUN'}))
PY

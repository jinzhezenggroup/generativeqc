#!/usr/bin/env bash
# Run under one finite srun allocation; arguments name clean, built checkouts.
set -euo pipefail
baseline_root=$(realpath "$1")
candidate_root=$(realpath "$2")
output_root=$(realpath -m "$3")
runner_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run this script through srun}"
: "${CUDA_VISIBLE_DEVICES:?Preserve Slurm device visibility}"
mkdir -p "$output_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GENERATIVEQC_PROFILE=off
export R1_OUTPUT_ROOT="$output_root"
for order in ABBA BAAB; do
  for block in 0 1 2 3; do
    if [[ "${order:$block:1}" == A ]]; then
      variant=baseline
      cd "$baseline_root"
      export R1_SOURCE_REVISION=df2ded411074d3b04b5c236b83d760ca1aafc295
    else
      variant=current
      cd "$candidate_root"
      export R1_SOURCE_REVISION=685478ca1111407573a5cbfa813d63ae0a5713ce
    fi
    export PYTHONPATH=python:$PWD
    export GENERATIVEQC_LIBRARY=$PWD/build/native-ks-catalog/libgenerativeqc.so
    export R1_SOURCE_MANIFEST="$output_root/source-inventory-$variant.json"
    # Authenticate measured inputs independently of the worktree's branch name.
    python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
revision = os.environ['R1_SOURCE_REVISION']
entries = subprocess.check_output(['git', 'ls-tree', '-rz', revision, '--',
    'CMakeLists.txt', 'cmake', 'include', 'src', 'python', 'tools', 'manifests', 'tests'])
files = {}
for entry in entries.split(b'\0'):
    if not entry:
        continue
    meta, raw_path = entry.split(b'\t', 1)
    _, kind, oid = meta.split()
    if kind != b'blob':
        continue
    path = raw_path.decode()
    expected = subprocess.check_output(['git', 'cat-file', 'blob', oid.decode()])
    assert Path(path).read_bytes() == expected, path
    files[path] = hashlib.sha256(expected).hexdigest()
Path(os.environ['R1_SOURCE_MANIFEST']).write_text(json.dumps(
    {'revision': revision, 'files': files}, sort_keys=True) + '\n')
PY
    python "$runner_root/endpoint_benchmark.py" --samples 11 --warmups 3 \
      --output "$output_root/paired-$order-$block-$variant.jsonl"
  done
done
python "$runner_root/audit_paired.py" "$output_root" --output "$output_root/audit.json"

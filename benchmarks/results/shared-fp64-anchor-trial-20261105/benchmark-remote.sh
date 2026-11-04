#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run in a finite Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Preserve assigned visibility}"
root=/home/jzzeng/generativeqc-qualification/standard-fast-unified-anchor-20261105-c511f9a33
export PATH="$root/.venv/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/source/python:$root/source"
export GENERATIVEQC_LIBRARY="$root/source/build/libgenerativeqc.so"
export GENERATIVEQC_KS_RESIDENT_MAX_BYTES=2147483648
export EXPERIMENT_ROOT="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root/source"
scontrol show job "$SLURM_JOB_ID" -o > "$root/evidence/benchmark-slurm.txt"
python - "$root/evidence/source-authentication.json" <<'PY'
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from tools.generate_build_identity import _inventory, _source_identity
source = Path.cwd()
expected = _source_identity(source, _inventory(source, source / 'cmake/GenerativeQCSourceIdentity.json'))
library = ctypes.CDLL(os.environ['GENERATIVEQC_LIBRARY'])
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
observed = library.generativeqc_get_source_identity().decode()
assert expected == observed, (expected, observed)
driver = source / 'benchmarks/fast_anchor_default/endpoint.py'
with Path(sys.argv[1]).open('x') as stream:
    json.dump({'source_identity': observed, 'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
               'slurm_job_id': os.environ['SLURM_JOB_ID'], 'visibility': os.environ['CUDA_VISIBLE_DEVICES']}, stream, indent=2)
PY
status=0
for atoms in 24 48; do
  for policy in fp64 fast; do
    for round in 0 1 2; do
      order='0 1'
      if [[ $round == 1 ]]; then order='1 0'; fi
      for selection in $order; do
        output="$root/evidence/samples/water-$atoms-$policy-r$round-$selection"
        mkdir -p "$output"
        export GENERATIVEQC_KS_FIXED_ANCHOR_DELTA="$selection"
        date --iso-8601=ns > "$output/started.txt"
        result=0
        timeout --kill-after=20 900 python -m benchmarks.fast_anchor_default.endpoint \
          --atoms "$atoms" --policy "$policy" --delta "$selection" --warm-repeats 3 \
          --output "$output/result.json" > "$output/endpoint.log" 2>&1 || result=$?
        printf '%s\n' "$result" > "$output/exit-code.txt"
        date --iso-8601=ns > "$output/completed.txt"
        printf 'water-%s %s round=%s selector=%s exit=%s\n' "$atoms" "$policy" "$round" "$selection" "$result"
        tail -n 1 "$output/endpoint.log"
        if [[ $result != 0 ]]; then status=1; fi
      done
    done
  done
done
exit "$status"

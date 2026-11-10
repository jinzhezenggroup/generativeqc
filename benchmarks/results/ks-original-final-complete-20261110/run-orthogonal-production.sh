#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/qc-ks-orthogonal-production-20261110
export PATH=$PWD/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH=python:.:tests/python:$PWD/.reference-deps
export LD_LIBRARY_PATH=$PWD:$PWD/blas:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GENERATIVEQC_KS_STATE_CUDA_TEST=1 GENERATIVEQC_GRID_CUDA_TEST=1
export GENERATIVEQC_LIBRARY=$PWD/candidate-production.so
unset GENERATIVEQC_DF_PROGRESS_TRACE
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
mkdir -p full-original-gates
printf 'HOST=%s JOB=%s STEP=%s VISIBLE=%s ARM=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_STEP_ID:-}" "$CUDA_VISIBLE_DEVICES" "$1"
cat source-revision.txt production-build-identity.json
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv
sha256sum "$GENERATIVEQC_LIBRARY" src/dft/{cuda_ks.cpp,cuda_ks_diis_policy.hpp} python/generativeqc/{ks.py,calculator.py,_ks_snapshot.py,_stationary_cuda.py}
/home/jzzeng/miniconda3/bin/python - "$1" <<'PY'
import ctypes
import json
import os
import sys
from pathlib import Path
from generativeqc.profiles import probe_device

record = probe_device(ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"]))
assert record["device"]["release_build"] == 1
assert record["device"]["fast_compile"] == 0
assert record["source_identity"] == json.loads(Path("production-build-identity.json").read_text())["source_identity"]
Path(f"full-original-gates/device-{sys.argv[1]}.json").write_text(json.dumps(record, indent=2))
print("DEVICE_PROBE=" + json.dumps(record, sort_keys=True))
PY
sccache --version
sccache --show-stats
status=0
run_gate() {
    local name=$1
    shift
    if "$@"; then
        printf '%s_exit=0 %s\n' "$name" "$(date -Iseconds)"
    else
        local result=$?
        printf '%s_exit=%s %s\n' "$name" "$result" "$(date -Iseconds)"
        status=1
    fi
}
if [[ "$1" == endpoints ]]; then
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/rsh-production
    run_gate original_rsh timeout 1800 /home/jzzeng/miniconda3/bin/python -u -m pytest -q tests/python/test_ks_original_rsh_admission_cuda.py --junitxml=full-original-gates/rsh-production.xml
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/uks-production
    run_gate original_uks timeout 1800 /home/jzzeng/miniconda3/bin/python -u -m pytest -q tests/python/test_ks_scientific_gates_cuda.py --junitxml=full-original-gates/uks-production.xml
    run_gate native_df_lr timeout 600 ./generativeqc_ks_cuda_tests --df-range-exchange
else
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/water-production-$1
    run_gate original_water_$1 timeout 7000 /home/jzzeng/miniconda3/bin/python -u -m pytest -q -s tests/python/test_ks_seed_admission_cuda.py -k "original_grid and $1" --junitxml=full-original-gates/water-production-$1.xml
fi
sccache --show-stats
exit "$status"

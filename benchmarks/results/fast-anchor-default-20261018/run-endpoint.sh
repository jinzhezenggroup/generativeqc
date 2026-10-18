#!/usr/bin/env bash
# Run inside a finite Slurm allocation after sourcing the CUDA/Python toolchain.
set -euo pipefail
: "${SLURM_JOB_ID:?Finite Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Preserve scheduler visibility}"
: "${EXPERIMENT_ROOT:?Source-authenticated qualification directory required}"
: "${PYTHON:?Qualified Python executable or runtime wrapper required}"
policy=${1:?fp64 or fast}
atoms=${2:?atom count}
selection=${3:?auto, 0 or 1}
output=${4:?new receipt directory}
export GENERATIVEQC_LIBRARY="$EXPERIMENT_ROOT/library/libgenerativeqc.so"
export PYTHONPATH="$EXPERIMENT_ROOT/source:$EXPERIMENT_ROOT/source/python"
export LD_LIBRARY_PATH="$EXPERIMENT_ROOT/library:${LD_LIBRARY_PATH:-}"
unset GENERATIVEQC_EXPERIMENT_ANCHOR_DELTA GENERATIVEQC_EXPERIMENT_VV10_INNER_POLICY
unset GENERATIVEQC_VV10_NATIVE_SEED GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED
unset GENERATIVEQC_INCREMENTAL_DIRECT_JK GENERATIVEQC_KS_FAST_ANCHOR_DELTA
if [[ $selection != auto ]]; then
    export GENERATIVEQC_KS_FAST_ANCHOR_DELTA="$selection"
fi
mkdir "$output"
date --iso-8601=ns > "$output/started.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$output/slurm.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$output/gpu.txt"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$output/visibility.txt"
sha256sum "$GENERATIVEQC_LIBRARY" > "$output/library-before.sha256"
launcher=$(command -v sccache || command -v ccache)
"$launcher" --version > "$output/compiler-cache-version.txt"
"$launcher" --show-stats > "$output/compiler-cache-before.txt"
cd "$EXPERIMENT_ROOT/source"
"$PYTHON" - > "$output/source-authentication.json" <<'PY'
import ctypes
import json
import os
from pathlib import Path
from tools.generate_build_identity import _inventory, _source_identity
source = Path.cwd()
expected = _source_identity(source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json"))
library = ctypes.CDLL(os.environ["GENERATIVEQC_LIBRARY"])
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
observed = library.generativeqc_get_source_identity().decode()
assert expected == observed, (expected, observed)
print(json.dumps({"native_identity": observed, "source": str(source)}))
PY
limit=1200
if [[ $atoms == 96 ]]; then limit=6600; fi
result=0
timeout --kill-after=30 "$limit" "$PYTHON" "$EXPERIMENT_ROOT/endpoint.py" \
    --atoms "$atoms" --policy "$policy" --delta "$selection" \
    --output "$output/result.json" > "$output/endpoint.log" 2>&1 || result=$?
printf '%s\n' "$result" > "$output/exit-code.txt"
sha256sum "$GENERATIVEQC_LIBRARY" > "$output/library-after.sha256"
"$launcher" --show-stats > "$output/compiler-cache-after.txt"
date --iso-8601=ns > "$output/completed.txt"
tail -1 "$output/endpoint.log"
exit "$result"

#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?finite allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve assigned visibility}"
root=/data/jzzeng/gqc-direct-md-20261016
arm=${1:?source arm}
method=${2:-r2scan-3c}
atoms=${3:-96}
trial=${4:-first}
operation=${5:-endpoint}
directory="$root/results/$arm/$operation/$method/$atoms/$trial"
mkdir -p "$directory"
cd "$root/$arm/source"
export PATH=/data/jzzeng/module-fast-fock-20261015/tools:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export GENERATIVEQC_LIBRARY="$root/$arm/runtime/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$root/$arm/stationary-cache"
export LD_LIBRARY_PATH="$root/$arm/runtime:/data/jzzeng/module-fast-fock-20261015/tools:/group/software/cuda-12.9.1/lib64"
export PYTHONPATH=.:python:tests/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python=/data/jzzeng/qc-r2scan-force-branches-20261015/python.sh
references=/data/jzzeng/gqc-direct-next-20261016
scontrol show job "$SLURM_JOB_ID" -o > "$directory/job.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv > "$directory/device.txt"
sha256sum "$GENERATIVEQC_LIBRARY" > "$directory/library.sha256"
ccache --version > "$directory/cache-version.txt"
ccache --show-stats > "$directory/cache-before.txt"
basis="$references/def2-svp.json"
[[ "$method" == r2scan-3c ]] && basis=python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json
arguments=(--method "$method" native --atoms "$atoms" --basis-file "$basis"
    --density-fitting none --repeats 5
    --reference "$references/references/none/$method/$atoms/reference.json"
    --output "$directory/native.json")
if [[ "$method" == hf ]]; then
    # Historical HF rows retain independent complete E/F but are not a newly
    # measured reference timing population. Require both geometry oracles;
    # the public endpoint still enforces every ordinary E/F accuracy gate.
    "$python" - "$references/references/none/$method/$atoms/reference.json" <<'PY'
import json, sys
from benchmarks.readme_omol25 import reference_baselines
record = json.load(open(sys.argv[1]))
assert len(reference_baselines(record, record["protocol"], allow_missing=True)) == 2
PY
    arguments+=(--allow-missing-reference)
fi
if [[ "$operation" == endpoint ]]; then
    timeout --kill-after=10 1800 "$python" "$references/with-native-work.py" "${arguments[@]}"
elif [[ "$operation" == profile ]]; then
    export GENERATIVEQC_DF_TRACE="$directory/events.jsonl"
    export GENERATIVEQC_DIRECT_FORCE_CENSUS="$directory/census.jsonl"
    export GENERATIVEQC_DIRECT_FORCE_CENSUS_SPLIT_LOW=1
    export GENERATIVEQC_MD_J_WORK_COUNTS=1
    nsys=/group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys
    timeout --kill-after=10 1800 "$nsys" profile --sample=none --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
        --output="$directory/cold-warm" "$python" -m benchmarks.profile_dft_endpoint \
        --capture all "${arguments[@]}"
    "$nsys" export --type sqlite --force-overwrite=true \
        --output="$directory/cold-warm.sqlite" "$directory/cold-warm.nsys-rep"
else
    exit 2
fi
ccache --show-stats > "$directory/cache-after.txt"
date --iso-8601=ns

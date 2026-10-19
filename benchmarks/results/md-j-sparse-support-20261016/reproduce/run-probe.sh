#!/usr/bin/env bash
set -euo pipefail
# Post-measurement safety repair; measured scripts remain in the immutable ef02 archive.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
safety="$script_dir/output_safety.py"
: "${SLURM_JOB_ID:?finite Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
root=${GQC_MD_REPRO_ROOT:-/data/jzzeng/gqc-direct-md-20261016}
if [[ "$root" != /* ]]; then
    echo "GQC_MD_REPRO_ROOT must be an absolute path" >&2
    exit 2
fi
arm=${1:?source arm required}
operation=${2:-checks}
trial=${3:-first}
directory="$root/results/$arm/$operation/$trial"
python3 "$safety" directory "$directory"
finish() {
    local original_status=$?
    trap - EXIT
    set +e
    ccache --show-stats > "$directory/cache-after.txt" 2>&1
    python3 "$safety" status "$directory/exit-status.json" "$original_status" >&2
    exit "$original_status"
}
trap finish EXIT

cd "$root/$arm/source"
export SOURCE="$PWD"
export PATH=/data/jzzeng/module-fast-fock-20261015/tools:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export GENERATIVEQC_LIBRARY="$root/$arm/runtime/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$root/$arm/stationary-cache"
export LD_LIBRARY_PATH="$root/$arm/runtime:/data/jzzeng/module-fast-fock-20261015/tools:/group/software/cuda-12.9.1/lib64"
export PYTHONPATH=.:python:tests/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python=/data/jzzeng/qc-r2scan-force-branches-20261015/python.sh
scontrol show job "$SLURM_JOB_ID" -o > "$directory/job.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv > "$directory/device.txt"
sha256sum "$GENERATIVEQC_LIBRARY" > "$directory/library.sha256"
ccache --version > "$directory/cache-version.txt"
ccache --show-stats > "$directory/cache-before.txt"
if [[ "$operation" == checks ]]; then
    export GENERATIVEQC_TEST_MD_J_NORMAL=1
    timeout --kill-after=10 600 "$python" -m pytest -q tests/python/test_md_j_normal_cuda.py
elif [[ "$operation" == sanitizers ]]; then
    export GENERATIVEQC_TEST_MD_J_NORMAL=1
    for tool in memcheck synccheck racecheck; do
        echo "BEGIN_SANITIZER $tool"
        bound=420
        [[ "$tool" != racecheck ]] || bound=900
        timeout --kill-after=10 "$bound" compute-sanitizer --tool "$tool" --target-processes all \
            --error-exitcode=1 "$python" -m pytest -q tests/python/test_md_j_normal_cuda.py
        echo "END_SANITIZER $tool"
    done
elif [[ "$operation" == diagnostic ]]; then
    export DIAGNOSTIC_ARM="$arm"
    export DIAGNOSTIC_LABEL="$operation/$trial/raw"
    timeout --kill-after=10 1800 "$python" "$script_dir/resident-diagnostic.py"
else
    echo "unsupported probe operation: $operation" >&2
    exit 2
fi

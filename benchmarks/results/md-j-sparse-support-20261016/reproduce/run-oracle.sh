#!/usr/bin/env bash
set -euo pipefail
# Post-measurement safety repair; measured scripts remain in the immutable ef02 archive.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
safety="$script_dir/output_safety.py"
: "${SLURM_JOB_ID:?finite allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve visibility}"
root=${GQC_MD_REPRO_ROOT:-/data/jzzeng/gqc-direct-md-20261016}
if [[ "$root" != /* ]]; then
    echo "GQC_MD_REPRO_ROOT must be an absolute path" >&2
    exit 2
fi
trial=${1:-first}
directory="$root/results/oracle/$trial"
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
export GQC_MD_ORACLE_OUTPUT="$directory/selected-j-oracle.json"
cd "$root/candidate-current/source"
export SOURCE="$PWD"
export PATH=/data/jzzeng/module-fast-fock-20261015/tools:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export GENERATIVEQC_LIBRARY="$root/candidate-current/runtime/libgenerativeqc.so"
export LD_LIBRARY_PATH="$root/candidate-current/runtime:/data/jzzeng/module-fast-fock-20261015/tools:/group/software/cuda-12.9.1/lib64"
export PYTHONPATH=.:python:tests/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
scontrol show job "$SLURM_JOB_ID" -o
sha256sum "$script_dir/selected-j-oracle.py" "$GENERATIVEQC_LIBRARY"
timeout --kill-after=10 840 /data/jzzeng/qc-r2scan-force-branches-20261015/python.sh "$script_dir/selected-j-oracle.py"

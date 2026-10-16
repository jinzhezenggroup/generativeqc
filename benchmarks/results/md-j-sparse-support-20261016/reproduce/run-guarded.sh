#!/usr/bin/env bash
set -euo pipefail
# Post-measurement safety repair; measured scripts remain in the immutable ef02 archive.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
safety="$script_dir/output_safety.py"
: "${SLURM_JOB_ID:?finite allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
root=${GQC_MD_REPRO_ROOT:-/data/jzzeng/gqc-direct-md-20261016}
if [[ "$root" != /* ]]; then
    echo "GQC_MD_REPRO_ROOT must be an absolute path" >&2
    exit 2
fi
receipt="$root/results/guards/$SLURM_JOB_ID/${1:?arm}/${2:-r2scan-3c}/${3:-96}/${4:-first}/${5:-endpoint}"
python3 "$safety" directory "$receipt"
child=
cleanup() {
    local original_status=$?
    trap - EXIT
    set +e
    if [[ -n "$child" ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$receipt/after.csv"
    observation_status=$?
    if [[ "$original_status" == 0 ]] && { [[ "$observation_status" != 0 ]] || [[ $(wc -l < "$receipt/after.csv") -gt 1 ]]; }; then
        original_status=1
    fi
    python3 "$safety" status "$receipt/exit-status.json" "$original_status" >&2
    exit "$original_status"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$receipt/before.csv"
if [[ $(wc -l < "$receipt/before.csv") -gt 1 ]]; then
    cat "$receipt/before.csv" >&2
    echo "Refusing qualification with an existing compute context" >&2
    exit 1
fi
timeout --kill-after=10 1860 bash "$script_dir/run-endpoint.sh" "$@" &
child=$!
wait "$child"
child=

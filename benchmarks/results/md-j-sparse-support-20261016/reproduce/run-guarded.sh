#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?finite allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
root=/data/jzzeng/gqc-direct-md-20261016
receipt="$root/results/guards/$SLURM_JOB_ID"
mkdir -p "$receipt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$receipt/before.csv"
if [[ $(wc -l < "$receipt/before.csv") -gt 1 ]]; then
    cat "$receipt/before.csv" >&2
    echo "Refusing qualification with an existing compute context" >&2
    exit 1
fi
child=
cleanup() {
    if [[ -n "$child" ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$receipt/after.csv"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
timeout --kill-after=10 1860 bash "$root/run-endpoint.sh" "$@" &
child=$!
wait "$child"
child=

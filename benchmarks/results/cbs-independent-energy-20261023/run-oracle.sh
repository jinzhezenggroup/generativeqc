#!/usr/bin/env bash
# Run in an assigned CPU instance; checkpoints survive this finite process cap.
set -euo pipefail
root="${1:?absolute task root}"
carbon="${2:?QM7 carbon count}"
threads="${3:?allocated numeric threads}"
memory_mb="${4:?PySCF memory limit in MB}"
output="${5:?fresh attempt output directory}"
[[ "$root" == /inspire/* && "$output" == /inspire/* ]]
cd "$root"
source /opt/generativeqc-reuse/activate.sh
export PYTHONPATH=/opt/generativeqc-reuse/venv/lib/python3.11/site-packages
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export TMPDIR="$root/tmp-C$carbon"
mkdir -p "$TMPDIR" "$output"
exec 9>"$root/C$carbon.lease"
flock -n 9 || exit 73
# Opening stdout would already truncate a prior attempt before Python can refuse.
# The molecule lease serializes this check with every invocation of this wrapper.
for artifact in "$output/C$carbon."* "$output/C$carbon-"*; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then
    printf 'Refusing existing molecule output: %s; use a fresh attempt directory\n' "$artifact" >&2
    exit 75
  fi
done
set +e
timeout --kill-after=30 21600 "$root/env/bin/python" "$root/oracle.py" \
  --inputs "$root/inputs.json" --dataset "$root/qm7.mat" --output "$output" \
  --carbon "$carbon" --threads "$threads" --memory-mb "$memory_mb" \
  > "$output/C$carbon.stdout.log" 2>&1
code=$?
printf '{"exit_code":%s,"time_limit_seconds":21600}\n' "$code" > "$output/C$carbon.outcome.json"
exit "$code"

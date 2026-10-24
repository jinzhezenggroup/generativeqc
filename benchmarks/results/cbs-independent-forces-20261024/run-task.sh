#!/usr/bin/env bash
# One output namespace is claimed before any log redirect; retries retain bytes.
set -euo pipefail
root="${1:?absolute frozen root}"
task="${2:?task id}"
threads="${3:?allocated threads}"
memory_mb="${4:?PySCF MB}"
[[ "$root" == /inspire/* ]]
[[ "$task" =~ ^C(4|7)-h(0|1)-(minus|plus)$ ]]
cd "$root"
source /opt/generativeqc-reuse/activate.sh
export PYTHONPATH=/opt/generativeqc-reuse/venv/lib/python3.11/site-packages
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
exec 9>"$root/$task.lease"
flock -n 9 || exit 73
output="$root/outputs/$task"
mkdir -p "$root/outputs"
# mkdir without -p is atomic and rejects any existing directory/file/symlink.
mkdir "$output" || exit 75
export TMPDIR="$output/tmp"
mkdir "$TMPDIR"
set +e
timeout --kill-after=30 21600 "$root/env/bin/python" "$root/oracle.py" \
  --inputs "$root/inputs.json.xz" --dataset "$root/qm7.mat" --task "$task" \
  --output "$output/science" --threads "$threads" --memory-mb "$memory_mb" \
  > "$output/stdout.log" 2>&1
code=$?
printf '{"exit_code":%s,"time_limit_seconds":21600}\n' "$code" > "$output/outcome.json"
"$root/env/bin/python" - "$output" <<'PY'
import json
import stat
import sys
from pathlib import Path
root = Path(sys.argv[1])
names = ('stdout.log', 'outcome.json', 'science/receipt.json',
         'science/basis-2.log', 'science/basis-3.log')
modes = {name: format(stat.S_IFREG | stat.S_IMODE((root / name).stat().st_mode), '06o')
         for name in names if (root / name).is_file()}
(root / 'modes.json').write_text(json.dumps(modes, indent=2) + '\n')
PY
exit "$code"

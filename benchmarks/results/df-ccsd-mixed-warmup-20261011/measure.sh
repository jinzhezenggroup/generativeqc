#!/usr/bin/env bash
set -euo pipefail
root=$1
pairs=${2:-1}
first_pair=${3:-1}
export PATH="/data/jzzeng/issue1972-20261005/deps/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="/data/jzzeng/issue1972-20261005/deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/candidate/python:$root/candidate:/data/jzzeng/issue1972-20261005/deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
output="$root/runs-$SLURM_JOB_ID"
mkdir -p "$output"
cp "$0" "$output/reproduce.sh"
printf '%s\n' "job=$SLURM_JOB_ID" "host=$(hostname)" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > "$output/allocation.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,power.limit --format=csv > "$output/gpu.csv"
sha256sum "$root/master/libgenerativeqc.so" "$root/master/endpoint" \
  "$root/candidate/build-cuda/libgenerativeqc.so" "$root/candidate/build-cuda/endpoint" \
  "$root/ethane230.input" > "$output/binaries.sha256"
cd "$root/candidate"
export GENERATIVEQC_LIBRARY="$PWD/build-cuda/libgenerativeqc.so"
export GENERATIVEQC_DF_CC_CUDA_TEST=1 GENERATIVEQC_DF_CC_USE_LIBRARY=1
export GENERATIVEQC_DF_LAMBDA_CUDA_TEST=1
export GENERATIVEQC_DF_MIXED_ENDPOINT_TEST=1
export GENERATIVEQC_DF_FORCE_ENDPOINT_BINARY="$PWD/build-cuda/endpoint"
ccache --version > "$output/ccache-version.txt"
ccache --show-stats > "$output/ccache-before.txt"
python -m pytest -q -rs --basetemp="$output/pytest" tests/python/test_df_cc_mixed_precision.py \
  tests/python/test_df_lambda_matrix_audit.py \
  tests/python/test_df_triples_mixed_endpoint.py \
  -k 'not complete_mixed_force_has_independent_energy_and_force_gates' \
  --maxfail=1 > "$output/tests.log" 2>&1
ccache --show-stats > "$output/ccache-after.txt"
controls=(1 1 1 1 32 8 8 0 0 2 1 30 0 0 1 auto 1 0 auto 0 0 30 1 1 1)
printf '%s\n' "${controls[*]}" > "$output/controls.txt"
for pair in $(seq "$first_pair" "$((first_pair + pairs - 1))"); do
  modes=(master mixed)
  if ((pair % 2 == 0)); then modes=(mixed master); fi
  for mode in "${modes[@]}"; do
    name="$mode-$pair"
    binary="$root/master/endpoint"
    arguments=()
    if [[ "$mode" == mixed ]]; then
      binary="$root/candidate/build-cuda/endpoint"
      arguments=(1)
    fi
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
      --format=csv --loop-ms=200 > "$output/$name.gpu.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    printf '%s %s started\n' "$(date --iso-8601=seconds)" "$name" >> "$output/progress.txt"
    env -u GENERATIVEQC_DF_TRACE -u GENERATIVEQC_DF_PROGRESS_TRACE \
      /usr/bin/time -v -o "$output/$name.time.txt" \
      "$binary" "$root/ethane230.input" "$output/$name.json" \
      "${controls[@]}" "${arguments[@]}" > "$output/$name.log" 2>&1
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    trap - EXIT
    printf '%s %s complete\n' "$(date --iso-8601=seconds)" "$name" >> "$output/progress.txt"
  done
  python - "$output/master-$pair.json" "$output/mixed-$pair.json" <<'PY'
import json
import sys
from pathlib import Path

master, mixed = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
if mixed["native_seconds"] > 1.03 * master["native_seconds"]:
    raise SystemExit("stop paired qualification: complete endpoint regresses by more than 3%")
PY
done
printf '%s\n' complete > "$output/status.txt"

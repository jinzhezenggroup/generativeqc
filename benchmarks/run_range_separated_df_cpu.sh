#!/usr/bin/env bash
# Run the existing eight CPU workloads into a fresh, exclusively owned campaign.
set -euo pipefail
set -C
root=$(git rev-parse --show-toplevel)
cd "$root"
export PATH="$root/.venv/bin:$PATH"
export PYTHONPATH="$root/python:$root/tests/python:$root${PYTHONPATH:+:$PYTHONPATH}"
export GENERATIVEQC_LIBRARY="$root/build/cpu/libgenerativeqc.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${RANGE_DF_EVIDENCE_DIRECTORY:?Choose a new ignored campaign directory}"
: "${RANGE_DF_HARNESS_DIRECTORY:?Restore the immutable historical report/oracle harness}"
evidence=$RANGE_DF_EVIDENCE_DIRECTORY
harness=$RANGE_DF_HARNESS_DIRECTORY
mkdir -p "$(dirname "$evidence")"
mkdir "$evidence"
exec 3> "$evidence/campaign-progress.jsonl"
stage=setup
printf '{"stage":"%s","status":"running"}\n' "$stage" >&3
campaign_failed() {
  status=$?
  printf '{"stage":"%s","status":"failed","exit_code":%s}\n' "$stage" "$status" >&3
  exit "$status"
}
trap campaign_failed ERR
mkdir "$evidence/raw"
for basis in def2-svp def2-tzvp; do
  for endpoint in energy forces; do
    for selector in direct df; do
      name=cpu-water-${basis}-cc-pvdz-jkfit-${selector}-${endpoint}
      stage=endpoint-$name
      printf '{"stage":"%s","status":"running"}\n' "$stage" >&3
      python -m benchmarks.range_separated_df --device cpu --selector "$selector" \
        --basis "$basis" --auxiliary cc-pvdz-jkfit --endpoint "$endpoint" \
        --repeats 3 --geometry-repeats 3 --source-head "$(git rev-parse HEAD)" \
        --build-boundary 'Release scalar CPU; stable whitened response' \
        --output "$evidence/raw/$name.json" > "$evidence/raw/$name.log" 2>&1
    done
  done
done
stage=report
printf '{"stage":"%s","status":"running"}\n' "$stage" >&3
python "$harness/report.py" "$evidence"/raw/cpu-*.json --output "$evidence/summary-cpu.json" \
  > "$evidence/summary-cpu.log" 2>&1
stage=oracle
printf '{"stage":"%s","status":"running"}\n' "$stage" >&3
python "$harness/../range-separated-df-stable-20261015/oracle-all.py" \
  "$evidence"/raw/cpu-*.json --output "$evidence/oracle-all-cpu.json" \
  > "$evidence/oracle-all-cpu.log" 2>&1
printf '{"stage":"complete","status":"complete"}\n' >&3

#!/usr/bin/env bash
set -euo pipefail
variant="$1"
carbon="$2"
policy="$3"
name="$4"
source "/data/jzzeng/qc-cbs-fast-memory-20261021/$variant/environment.sh"
export GENERATIVEQC_PROFILE=off
output="$CBS_FAST_ROOT/results/$name"
mkdir -p "$output"
result=0
timeout --kill-after=15 3600 bash "$CBS_FAST_ROOT/python-preload.sh" -m benchmarks.readme_cbs native \
  --qm7 /data/jzzeng/readme-cbs-20261013/qm7.mat --carbon "$carbon" \
  --policy "$policy" --hf-reference df --budget-mib 32768 --samples 1 \
  --source-revision "976f756b3-$variant-gmres-capacity" \
  --output "$output/native.json" > "$output/native.log" 2>&1 || result=$?
printf '{"exit_code":%s,"point_time_limit_seconds":3600,"slurm_job_id":"%s"}\n' "$result" "$SLURM_JOB_ID" > "$output/native.outcome"
exit "$result"

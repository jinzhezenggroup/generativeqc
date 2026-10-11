#!/usr/bin/env bash
# #1598 Incremental Direct-J/K gate: complete warm E/F, prepared-plan cold.
#
# Runs the audited HF comparator with independent GPU4PySCF references over
# RHF, UHF-cation and moved nested-water populations at 96/192/384/768 AOs,
# interleaved A-B-B-A per size, cold plus fixed-density warm replays per run,
# plus a PBE0 cold re-verification control. Every run is a fresh process.
set -euo pipefail

require_clean_checkout() {
  local state
  state="$(git status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$state" ]]; then
    printf 'Qualification requires a clean checkout, including untracked inputs\n' >&2
    return 2
  fi
}
require_clean_checkout

: "${GENERATIVEQC_LIBRARY:?Select the Release CUDA library built from the measured source}"
: "${ISSUE1598_ALLOCATION:?Name the finite GPU allocation (Slurm job id or notebook name)}"

python="${README_BENCHMARK_PYTHON:-python}"
output="${ISSUE1598_BENCHMARK_OUTPUT:-.artifacts/issue1598-incremental-direct-jk}"
point_timeout="${ISSUE1598_POINT_TIMEOUT:-2400}"
rebuild_interval="${ISSUE1598_REBUILD_INTERVAL:-8}"
[[ "$point_timeout" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid point timeout\n' >&2; exit 2; }
[[ "$rebuild_interval" =~ ^[0-9]+$ ]] || { printf 'Invalid rebuild interval\n' >&2; exit 2; }

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export ISSUE1598_HOSTNAME="$(hostname)"
mkdir -p "$output"/{rhf,uhf,moved,ks,ledger}

git rev-parse HEAD > "$output/source-sha.txt"
sha256sum "$GENERATIVEQC_LIBRARY" > "$output/library-sha.txt"
{
  date --iso-8601=seconds
  echo "allocation=$ISSUE1598_ALLOCATION hostname=$(hostname)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} > "$output/device.txt"

ledger() {
  local point="$1"
  {
    date --iso-8601=seconds
    echo "point=$point"
    nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,power.draw --format=csv,noheader
  } >> "$output/ledger/nvidia-smi.tsv"
}

run_point() {
  local population="$1" mode="$2" case_name="$3" atoms="$4" round="$5"
  local enabled=0
  [[ "$mode" == incremental ]] && enabled=1
  mkdir -p "$output/$population"
  local prefix="$output/$population/direct-$atoms-$mode-$round"
  local env_args=(GENERATIVEQC_INCREMENTAL_DIRECT_JK="$enabled")
  if [[ "$mode" == incremental ]]; then
    env_args+=(GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL="$rebuild_interval")
  fi
  printf 'RUN mode=%s case=%s atoms=%s round=%s\n' "$mode" "$case_name" "$atoms" "$round"
  local attempt
  for attempt in 1 2; do
    local attempt_prefix="$prefix-attempt$attempt"
    if timeout --kill-after=5 "$point_timeout" env "${env_args[@]}" "$python" \
        -m benchmarks.readme_hf_scaling_qualification --case "$case_name" \
        --batch 1 --repeats 3 --max-iterations 100 \
        --energy-tolerance 1e-10 --density-tolerance 1e-9 \
        --reference-gradient-tolerance 1e-8 --screening-tolerance 1e-12 \
        --maximum-energy-error 1e-8 --maximum-force-error 1e-7 \
        --allow-portable-build \
        --output "$attempt_prefix.json" \
        --progress-output "$attempt_prefix.progress.jsonl" \
        > "$attempt_prefix.log" 2>&1; then
      printf '{"exit_code":0,"attempt":%s}\n' "$attempt" > "$prefix.outcome"
      cp "$attempt_prefix.json" "$prefix.json"
      cp "$attempt_prefix.progress.jsonl" "$prefix.progress.jsonl"
      cp "$attempt_prefix.log" "$prefix.log"
      ledger "$case_name-$atoms-$mode-$round"
      return 0
    else
      status=$?
    fi
    printf '{"exit_code":%s,"attempt":%s}\n' "$status" "$attempt" > "$attempt_prefix.outcome"
    printf 'FAIL mode=%s case=%s attempt=%s exit=%s (see %s-attempt%s.log)\n' \
      "$mode" "$case_name" "$attempt" "$status" "$prefix" "$attempt" >&2
    ledger "fail-$case_name-$atoms-$mode-$round-attempt$attempt"
    if [[ "$attempt" == 1 ]]; then sleep 30; fi
  done
  printf '{"exit_code":%s,"attempt":2,"retried":true}\n' "$status" > "$prefix.outcome"
  return 0
}

# Interleaved A-B-B-A per size: each set of four runs shares the same
# allocation, binary, device visibility and local thermal/clock epoch.
for atoms in 12 24 48 96; do
  run_point rhf baseline "water-$atoms" "$atoms" 1
  run_point rhf incremental "water-$atoms" "$atoms" 1
  run_point rhf incremental "water-$atoms" "$atoms" 2
  run_point rhf baseline "water-$atoms" "$atoms" 2
done

for atoms in 12 24 48 96; do
  run_point uhf baseline "water-$atoms-cation-uhf" "$atoms" 1
  run_point uhf incremental "water-$atoms-cation-uhf" "$atoms" 1
  run_point uhf incremental "water-$atoms-cation-uhf" "$atoms" 2
  run_point uhf baseline "water-$atoms-cation-uhf" "$atoms" 2
done

for atoms in 24 96; do
  run_point moved baseline "water-$atoms-moved" "$atoms" 1
  run_point moved incremental "water-$atoms-moved" "$atoms" 1
  run_point moved incremental "water-$atoms-moved" "$atoms" 2
  run_point moved baseline "water-$atoms-moved" "$atoms" 2
done

# PBE0 cold re-verification control (fresh process per sample, off/on/on/off),
# independently re-checking the cautionary #1803-style pattern on this master.
ks_point() {
  local mode="$1" atoms="$2" round="$3"
  local enabled=0
  [[ "$mode" == on ]] && enabled=1
  local prefix="$output/ks/pbe0-$atoms-$mode-$round"
  printf 'RUN ks mode=%s atoms=%s round=%s\n' "$mode" "$atoms" "$round"
  local attempt
  for attempt in 1 2; do
    local attempt_prefix="$prefix-attempt$attempt"
    if timeout --kill-after=5 "$point_timeout" env \
        GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK="$enabled" \
        GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL=8 \
        GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD=0 \
        "$python" benchmarks/issue1598_ks_cold_control.py \
        "$attempt_prefix.json" "$atoms" \
        > "$attempt_prefix.log" 2>&1; then
      printf '{"exit_code":0,"attempt":%s}\n' "$attempt" > "$prefix.outcome"
      cp "$attempt_prefix.json" "$prefix.json"
      cp "$attempt_prefix.log" "$prefix.log"
      ledger "ks-$atoms-$mode-$round"
      return 0
    else
      status=$?
    fi
    printf '{"exit_code":%s,"attempt":%s}\n' "$status" "$attempt" > "$attempt_prefix.outcome"
    printf 'FAIL ks mode=%s atoms=%s attempt=%s exit=%s (see %s-attempt%s.log)\n' \
      "$mode" "$atoms" "$attempt" "$status" "$prefix" "$attempt" >&2
    ledger "fail-ks-$atoms-$mode-$round-attempt$attempt"
    if [[ "$attempt" == 1 ]]; then sleep 30; fi
  done
  printf '{"exit_code":%s,"attempt":2,"retried":true}\n' "$status" > "$prefix.outcome"
  return 0
}

ks_point off 96 1
ks_point on 96 1
ks_point on 96 2
ks_point off 96 2
ks_point off 48 1
ks_point on 48 1

"$python" tools/reduce_issue1598_incremental_qualification.py \
  --input "$output" --output "$output/summary.json"

#!/usr/bin/env bash
# Atomic one-shot launch; never redirect onto a previous suite's diagnostics.
set -euo pipefail
root="${1:?absolute root}"
threads="${2:?allocated threads}"
memory_mb="${3:?PySCF MB}"
[[ "$root" == /inspire/* ]]
cd "$root"
mkdir "$root/launch" || exit 75
nohup bash "$root/run-suite.sh" "$root" "$threads" "$memory_mb" \
  > "$root/launch/stdout.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$root/launch/pid"
printf 'suite_pid=%s\n' "$pid"

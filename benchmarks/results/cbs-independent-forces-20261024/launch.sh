#!/usr/bin/env bash
# Atomic one-shot launch; never redirect onto a previous suite's diagnostics.
set -euo pipefail
root="${1:?absolute root}"
threads="${2:?allocated threads}"
memory_mb="${3:?PySCF MB}"
[[ "$root" == /inspire/* ]]
cd "$root"
mkdir "$root/launch" || exit 75
nohup bash -c '
  set +e
  timeout --kill-after=30 180000 bash "$1/run-suite.sh" "$1" "$2" "$3"
  code=$?
  printf "{\"exit_code\":%s,\"controller_limit_seconds\":180000}\n" "$code" > "$1/launch/outcome.json"
  exit "$code"
' cbs-force-supervisor "$root" "$threads" "$memory_mb" \
  > "$root/launch/stdout.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$root/launch/pid"
printf 'suite_pid=%s\n' "$pid"

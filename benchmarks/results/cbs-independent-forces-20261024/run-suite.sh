#!/usr/bin/env bash
set -euo pipefail
root="${1:?absolute root}"
threads="${2:?allocated threads}"
memory_mb="${3:?PySCF MB}"
cd "$root"
exec 8>"$root/suite.lease"
flock -n 8 || exit 73
if [[ -e "$root/suite.outcome.json" || -L "$root/suite.outcome.json" || -e "$root/outputs" || -L "$root/outputs" ]]; then
  printf 'Refusing an existing suite output namespace\n' >&2
  exit 75
fi
# No automatic retries. Continue independent tasks while retaining any failures.
result=0
for carbon in 4 7; do
  for step in 0 1; do
    for sign in minus plus; do
      task="C$carbon-h$step-$sign"
      printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$task"
      if bash "$root/run-task.sh" "$root" "$task" "$threads" "$memory_mb"; then
        printf '%s completed %s\n' "$(date -u +%FT%TZ)" "$task"
      else
        code=$?
        printf '%s failed %s exit=%s\n' "$(date -u +%FT%TZ)" "$task" "$code"
        result=1
      fi
    done
  done
done
printf '{"exit_code":%s}\n' "$result" > "$root/suite.outcome.json"
exit "$result"

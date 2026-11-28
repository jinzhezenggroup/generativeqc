#!/usr/bin/env bash
set -euo pipefail
root=$1
run=$2
workers=32
per_process_bytes=$((6 * 1024 * 1024 * 1024))
date -u +%FT%TZ > "$run/entry.txt"
trap 'code=$?; date -u +%FT%TZ > "$run/exit.txt"; printf "%s\n" "$code" > "$run/exit-code.partial"; mv "$run/exit-code.partial" "$run/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$run/pid.txt"
printf 'Task: #3166 addendum aziridine PySCF finite differences\nSupervisor PID: %s\nWorkers: %s single-threaded, one Python coordinator\nAddress-space limit: 6 GiB per Python process, 198 GiB combined maximum\n' "$$" "$workers" > "$run/notebook-use.txt"
lscpu > "$run/cpu.txt"
grep -im1 '^model name.*XEON.*GOLD 6530' /proc/cpuinfo > "$run/validated-model.txt"
for control in cpu.max memory.max memory.current; do
  printf '%s %s\n' "$control" "$(cat "/sys/fs/cgroup/$control")"
done > "$run/cgroup.txt"
environment="$root/../issue-3166-20261125/cpu-env"
cp "$environment/packages.txt" "$run/packages.txt"
payload=$(mktemp -d "$run/payload.XXXXXXXX")
cp -r "$root/cpu-payload/." "$payload/"
(
  cd "$payload"
  sha256sum -c "$root/cpu-payload.sha256"
  patch --batch -p1 -i "$root/parallel-minima.patch"
  patch --batch -p1 -i "$root/resume-minima.patch"
)
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
prlimit --as="$per_process_bytes:$per_process_bytes" -- "$environment/venv/bin/python" -u \
  "$payload/check_df_minima.py" "$payload" "$root/native" "$run/raw" \
  --auxiliary "$payload/cc-pwcvtz-rifit.json" --molecule aziridine --workers "$workers" --resume &
driver=$!
printf 'Python coordinator PID: %s\n' "$driver" >> "$run/notebook-use.txt"
wait "$driver"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 ABSOLUTE_RESULT_DIRECTORY OBSERVER_COMMIT" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
result_directory=$1
observer_commit=$2
[[ $result_directory = /* ]] || usage
[[ $observer_commit =~ ^[0-9a-f]{40}$ ]] || usage
[[ ! -e $result_directory ]] || {
  echo "result directory already exists" >&2
  exit 3
}

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/../.." && pwd)
build_directory=$result_directory/build
umask 077
mkdir -p -- "$build_directory"

python=$(command -v python3 || true)
readelf=$(command -v readelf || true)
cxx=$(command -v "${CXX:-g++}" || true)
[[ -n $python && -n $readelf && -n $cxx ]] || {
  echo "python3, readelf and a C++ compiler are required" >&2
  exit 4
}

cuda_root=${CUDA_HOME:-/usr/local/cuda}
include_directory=$cuda_root/include
headers=(cuda_runtime_api.h cublas_v2.h cusolverDn.h)
for header in "${headers[@]}"; do
  [[ -f $include_directory/$header ]] || {
    echo "missing CUDA SDK header: $header" >&2
    exit 5
  }
done

version_at_least() {
  local observed=$1
  local required=$2
  [[ $(printf '%s\n%s\n' "$required" "$observed" | sort -V | head -n 1) == "$required" ]]
}

launcher=()
launcher_kind=
if command -v sccache >/dev/null 2>&1; then
  sccache_version=$(sccache --version | awk 'NR == 1 {print $2}')
  if [[ -n $sccache_version ]] && version_at_least "$sccache_version" 0.16.0; then
    launcher=(sccache)
    launcher_kind=sccache
  fi
fi
if [[ -z $launcher_kind ]] && command -v ccache >/dev/null 2>&1; then
  launcher=(ccache)
  launcher_kind=ccache
fi
[[ -n $launcher_kind ]] || {
  echo "no usable sccache >=0.16.0 or ccache launcher" >&2
  exit 6
}

if [[ $launcher_kind == sccache ]]; then
  sccache --version >"$result_directory/launcher-version.txt" 2>&1
  sccache --show-stats >"$result_directory/launcher-stats-before.txt" 2>&1
else
  ccache --version >"$result_directory/launcher-version.txt" 2>&1
  ccache --show-stats >"$result_directory/launcher-stats-before.txt" 2>&1
fi
"$cxx" --version >"$result_directory/compiler-version.txt" 2>&1

compile() {
  printf '%q ' "${launcher[@]}" "$cxx" "$@" >>"$result_directory/compile-commands.txt"
  printf '\n' >>"$result_directory/compile-commands.txt"
  "${launcher[@]}" "$cxx" "$@"
}

common=(-std=c++17 -O2 -Wall -Wextra -Werror -I"$include_directory")
compile "${common[@]}" -fPIC -shared "$script_directory/fake_provider.cpp" \
  -Wl,--version-script="$script_directory/fake_provider.map" \
  -Wl,-soname,libfake_rhf_provider.so -ldl \
  -o "$build_directory/libfake_rhf_provider.so"
compile "${common[@]}" "$script_directory/driver.cpp" \
  -L"$build_directory" -Wl,--no-as-needed -lfake_rhf_provider \
  '-Wl,-rpath,$ORIGIN' -ldl -o "$build_directory/fake_provider_driver"
compile "${common[@]}" "$script_directory/missing_driver.cpp" -ldl \
  -o "$build_directory/missing_provider_driver"
compile "${common[@]}" -fPIC -shared "$script_directory/fault_write.cpp" -ldl \
  -o "$build_directory/libfail_trace_write.so"
compile "${common[@]}" -fPIC -shared \
  "$repository_root/benchmarks/rhf_provider_retention_probe.cpp" \
  -Wl,--version-script="$repository_root/benchmarks/rhf_provider_retention_probe.map" \
  -ldl -o "$build_directory/librhf_provider_probe.so"

if [[ $launcher_kind == sccache ]]; then
  sccache --show-stats >"$result_directory/launcher-stats-after.txt" 2>&1
else
  ccache --show-stats >"$result_directory/launcher-stats-after.txt" 2>&1
fi

"$readelf" -WsW "$build_directory/fake_provider_driver" \
  >"$result_directory/driver.symbols.txt"
"$readelf" -WsW "$build_directory/libfake_rhf_provider.so" \
  >"$result_directory/fake_provider.symbols.txt"
"$readelf" -WsW "$build_directory/librhf_provider_probe.so" \
  >"$result_directory/shim.symbols.txt"
"$readelf" -V "$build_directory/libfake_rhf_provider.so" \
  >"$result_directory/fake_provider.versions.txt"
"$readelf" -V "$build_directory/fake_provider_driver" \
  >"$result_directory/driver.versions.txt"

sha256sum \
  "$repository_root/benchmarks/rhf_provider_retention_probe.cpp" \
  "$repository_root/benchmarks/rhf_provider_retention_probe.map" \
  "$repository_root/benchmarks/rhf_provider_retention_report.py" \
  "$script_directory/README.md" \
  "$script_directory/build_and_run.sh" \
  "$script_directory/driver.cpp" \
  "$script_directory/fake_provider.cpp" \
  "$script_directory/fake_provider.map" \
  "$script_directory/fault_write.cpp" \
  "$script_directory/missing_driver.cpp" \
  "$script_directory/verify_results.py" \
  >"$result_directory/source.sha256"
for header in "${headers[@]}"; do
  sha256sum "$include_directory/$header"
done >"$result_directory/headers.sha256"
sha256sum "$build_directory"/* >"$result_directory/binaries.sha256"

"$python" - "$result_directory/identity.json" "$observer_commit" "$launcher_kind" <<'PY'
import json
import sys
from pathlib import Path

path, observer_commit, launcher = sys.argv[1:]
identity = {
    "schema": "generativeqc.rhf-provider-fake-identity.v1",
    "runtime_source_commit": "a3088c5d2825de719123682facc36e8327a155e9",
    "production_library_sha256": "e39cbf740cd81ace24c56fd0c1e7bdb2bcaeb8dc93aca02c7f37c70d20c0e001",
    "production_endpoint_sha256": "19b34468267ef52174e14afb950cb80e08c33fd3d861b89b068819886120cc65",
    "observer_commit": observer_commit,
    "launcher": launcher,
    "scope": "fake ELF forwarding and observer protocol only; no CUDA device or science",
}
Path(path).write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

: >"$result_directory/case-exits.tsv"
record_run() {
  local name=$1
  shift
  set +e
  "$@" >"$result_directory/$name.stdout.json" 2>"$result_directory/$name.stderr.txt"
  local status=$?
  set -e
  printf '%s\t%s\n' "$name" "$status" >>"$result_directory/case-exits.tsv"
}

driver=$build_directory/fake_provider_driver
shim=$build_directory/librhf_provider_probe.so
fault=$build_directory/libfail_trace_write.so
missing=$build_directory/missing_provider_driver

record_run positive.baseline env FAKE_PROVIDER_CASE=positive "$driver"
record_run positive.shim env FAKE_PROVIDER_CASE=positive \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/positive.trace.jsonl" \
  LD_PRELOAD="$shim" "$driver"
record_run provider_failure.baseline env FAKE_PROVIDER_CASE=provider_failure "$driver"
record_run provider_failure.shim env FAKE_PROVIDER_CASE=provider_failure \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/provider_failure.trace.jsonl" \
  LD_PRELOAD="$shim" "$driver"
"$python" - "$result_directory/stdout-parity.json" \
  "$result_directory/positive.baseline.stdout.json" \
  "$result_directory/positive.shim.stdout.json" \
  "$result_directory/provider_failure.baseline.stdout.json" \
  "$result_directory/provider_failure.shim.stdout.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, positive_base, positive_shim, failure_base, failure_shim = sys.argv[1:]


def compare(baseline_name, shim_name):
    baseline = Path(baseline_name).read_bytes()
    shim = Path(shim_name).read_bytes()
    return {
        "baseline": {"size": len(baseline), "sha256": hashlib.sha256(baseline).hexdigest()},
        "shim": {"size": len(shim), "sha256": hashlib.sha256(shim).hexdigest()},
        "byte_identical": baseline == shim,
    }


value = {
    "positive": compare(positive_base, positive_shim),
    "provider_failure": compare(failure_base, failure_shim),
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
record_run overflow.shim env FAKE_PROVIDER_CASE=positive \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/overflow.trace.jsonl" \
  LD_PRELOAD="$shim" "$driver" --overflow
record_run resolution_failure.shim env \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/resolution_failure.trace.jsonl" \
  LD_PRELOAD="$shim" "$missing"
record_run o_excl.first env FAKE_PROVIDER_CASE=positive \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/o_excl.trace.jsonl" \
  LD_PRELOAD="$shim" "$driver"
if [[ -f $result_directory/o_excl.trace.jsonl ]]; then
  o_excl_sha_before=$(sha256sum "$result_directory/o_excl.trace.jsonl" | awk '{print $1}')
  o_excl_size_before=$(stat -c %s "$result_directory/o_excl.trace.jsonl")
  o_excl_mode=$(stat -c %a "$result_directory/o_excl.trace.jsonl")
else
  o_excl_sha_before=MISSING
  o_excl_size_before=-1
  o_excl_mode=MISSING
fi
record_run o_excl.second env FAKE_PROVIDER_CASE=positive \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/o_excl.trace.jsonl" \
  LD_PRELOAD="$shim" "$driver"
if [[ -f $result_directory/o_excl.trace.jsonl ]]; then
  o_excl_sha_after=$(sha256sum "$result_directory/o_excl.trace.jsonl" | awk '{print $1}')
  o_excl_size_after=$(stat -c %s "$result_directory/o_excl.trace.jsonl")
else
  o_excl_sha_after=MISSING
  o_excl_size_after=-1
fi
"$python" - "$result_directory/o_excl.meta.json" "$o_excl_mode" \
  "$o_excl_sha_before" "$o_excl_sha_after" "$o_excl_size_before" \
  "$o_excl_size_after" <<'PY'
import json
import sys
from pathlib import Path

path, mode, sha_before, sha_after, size_before, size_after = sys.argv[1:]
value = {
    "mode": mode,
    "sha256_before": sha_before,
    "sha256_after": sha_after,
    "size_before": int(size_before),
    "size_after": int(size_after),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
record_run write_failure.shim env FAKE_PROVIDER_CASE=positive \
  FAKE_FAIL_ONE_TRACE_WRITE=1 \
  GENERATIVEQC_RHF_PROVIDER_TRACE="$result_directory/write_failure.trace.jsonl" \
  LD_PRELOAD="$fault:$shim" "$driver"

fake_provider_sha=$(sha256sum "$build_directory/libfake_rhf_provider.so" | awk '{print $1}')
driver_sha=$(sha256sum "$driver" | awk '{print $1}')
shim_sha=$(sha256sum "$shim" | awk '{print $1}')
if [[ -f $result_directory/positive.trace.jsonl ]]; then
  trace_sha=$(sha256sum "$result_directory/positive.trace.jsonl" | awk '{print $1}')
else
  trace_sha=$(printf '0%.0s' {1..64})
fi
"$python" - "$result_directory/positive.policy.json" \
  "$result_directory/positive.report-identity.json" "$fake_provider_sha" \
  "$driver_sha" "$shim_sha" "$trace_sha" <<'PY'
import json
import sys
from pathlib import Path

policy_path, identity_path, provider_sha, driver_sha, shim_sha, trace_sha = sys.argv[1:]
source = "a3088c5d2825de719123682facc36e8327a155e9"
policy = {
    "source_commit": source,
    "providers": {
        "cublas": {
            "allowance_bytes": 100,
            "required_events": [
                "cublas_create",
                "cublas_set_stream",
                "cublas_set_pointer_mode",
            ],
        },
        "cusolver": {
            "allowance_bytes": 150,
            "required_events": [
                "cusolver_create",
                "cusolver_set_stream",
                "cusolver_create_params",
            ],
        },
    },
}
identity = {
    "source_commit": source,
    "library_sha256": provider_sha,
    "endpoint_sha256": driver_sha,
    "shim_sha256": shim_sha,
    "event_log_sha256": trace_sha,
    "device_uuid": "FAKE-CPU-ELF",
    "job_id": "issue1765-versioned-fake-provider",
}
Path(policy_path).write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
Path(identity_path).write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
PY
if [[ -f $result_directory/positive.trace.jsonl ]]; then
  set +e
  "$python" "$repository_root/benchmarks/rhf_provider_retention_report.py" \
    --events "$result_directory/positive.trace.jsonl" \
    --policy "$result_directory/positive.policy.json" \
    --identity "$result_directory/positive.report-identity.json" \
    --output "$result_directory/positive.report.json"
  report_status=$?
  set -e
else
  report_status=99
fi
printf '%s\n' "$report_status" >"$result_directory/positive.report.exit.txt"

set +e
"$python" "$script_directory/verify_results.py" "$result_directory"
verification_status=$?
set -e
if [[ $verification_status -eq 0 ]]; then
  echo "FAKE_PROVIDER_HARNESS_RESULT=PASS"
else
  echo "FAKE_PROVIDER_HARNESS_RESULT=FAIL"
fi
exit "$verification_status"

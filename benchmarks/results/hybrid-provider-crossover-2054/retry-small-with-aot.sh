#!/usr/bin/env bash
# Fresh clean-timing retry after the benchmark-only Direct metric guard fix.
set -euo pipefail

repo_root_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/issue-2054-crossover-20261008
core_source_2054=ed26480ecccef91506b411c773d4730760a04850
cd "$repo_root_2054"
: "${GENERATIVEQC_2054_FINITE_JOB:?finite Inspire GPU Job name required}"
test -z "$(git status --porcelain)"
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 1
if git diff --name-only "$core_source_2054" HEAD | grep -vE \
    '^(tools/benchmark_hybrid_provider_crossover.py|tests/python/test_hybrid_provider_crossover.py|benchmarks/results/hybrid-provider-crossover-2054/)'; then
  echo "scientific/build source differs from the frozen core build" >&2
  exit 1
fi

shared_bin_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin
cache_tool_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/bin/ccache"
cache_lib_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/lib/x86_64-linux-gnu"
build_dir_2054="$repo_root_2054/.artifacts/build-hybrid-2054-sm90-cuda129"
run_dir_2054="$repo_root_2054/.artifacts/benchmarks/hybrid-provider-2054/cuda129-moved-warm-retry"
mkdir -p "$run_dir_2054"
export PATH="$(dirname "$cache_tool_2054"):$shared_bin_2054:$PATH"
export LD_LIBRARY_PATH="$cache_lib_2054${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root_2054/.artifacts/python-site:$repo_root_2054/python:$repo_root_2054"
export PYTHONUTF8=1
export CCACHE_DIR="$repo_root_2054/.artifacts/compiler-cache/data"
export CCACHE_BASEDIR="$repo_root_2054"
export GENERATIVEQC_LIBRARY="$build_dir_2054/libgenerativeqc.so"
python_2054="$shared_bin_2054/python"
test -x "$cache_tool_2054" && test -x "$python_2054"

verify_sha_2054() {
  local expected_2054="$1" path_2054="$2"
  test "$(sha256sum "$path_2054" | cut -d' ' -f1)" = "$expected_2054"
}
verify_sha_2054 2e588e90b4e6b91ea31841e940e0ceb8e543ef7dfcf110785e614abb8ec492d4 "$GENERATIVEQC_LIBRARY"
verify_sha_2054 3a9dbe9fc6316811e4807e1115995d817067b146317f623146f546d766d9ecbb "$build_dir_2054/generativeqc_stationary_pbe0_rks.json"
verify_sha_2054 49d3c8d999ad8f00a99f6f2874ecd772ed8a01b74b7da53f4f5a11cd215e9044 "$build_dir_2054/libgenerativeqc_stationary_pbe0_rks.so"
verify_sha_2054 866839a68779637ce658e5cff1cde873620016883af7fd5bb6e7eae6738b70c7 "$build_dir_2054/generativeqc_stationary_pbe0_rks_spd.json"
verify_sha_2054 fab751d7ad76012213ba64d56513f2693c74c6e9b3b54ac5bf88292ce548fbd0 "$build_dir_2054/libgenerativeqc_stationary_pbe0_rks_spd.so"

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'inspire_job_name=%s\nrunner_head=%s\ncore_source=%s\n' \
    "$GENERATIVEQC_2054_FINITE_JOB" "$(git rev-parse HEAD)" "$core_source_2054"
  git diff --name-only "$core_source_2054" HEAD
  sha256sum "$GENERATIVEQC_LIBRARY" "$build_dir_2054"/generativeqc_stationary_pbe0_rks*.json "$build_dir_2054"/libgenerativeqc_stationary_pbe0_rks*.so
  "$cache_tool_2054" --version
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$run_dir_2054/receipt.txt" 2>&1

"$python_2054" tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-3 --output "$run_dir_2054/native"
"$python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
  --case water-3 --arm direct \
  --trace "$run_dir_2054/direct-trace.jsonl" \
  --output "$run_dir_2054/direct-profile.json"
"$python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
  --case water-3 --arm df-jk-occupied \
  --trace "$run_dir_2054/df-trace.jsonl" \
  --output "$run_dir_2054/df-profile.json"

"$python_2054" - "$run_dir_2054" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "native/campaign.json").read_text())
if manifest.get("status") != "RECORDED" or len(manifest.get("order", [])) != 5:
    raise SystemExit("#2054 ABBA smoke manifest is incomplete")
records = sorted(path for path in (root / "native").glob("*.json") if path.name != "campaign.json")
statuses = [(path.name, json.loads(path.read_text()).get("status")) for path in records]
for path in (root / "direct-profile.json", root / "df-profile.json"):
    value = json.loads(path.read_text())
    statuses.append((path.name, value.get("status")))
    if path.name == "df-profile.json" and not value.get("occupied_reuse_verified"):
        statuses.append(("occupied-reuse", "FAIL"))
print(statuses, flush=True)
if not statuses or any(status not in {"MEASURED", "UNSUPPORTED"} for _, status in statuses):
    raise SystemExit("#2054 moved-warm retry failed; inspect retained raw records")
PY

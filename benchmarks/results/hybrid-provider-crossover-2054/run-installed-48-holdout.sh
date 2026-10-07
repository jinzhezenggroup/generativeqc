#!/usr/bin/env bash
# Two supported arms, ABBA processes, 48-atom PBE0 and non-water holdout.
set -euo pipefail

repo_root_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/issue-2054-crossover-20261008
source_head_2054=2952481fccec5efcf3f993605b23ccefe01e640a
cd "$repo_root_2054"
: "${GENERATIVEQC_2054_FINITE_JOB:?finite Inspire GPU Job name required}"
test "$(git rev-parse HEAD)" = "$source_head_2054"
test -z "$(git status --porcelain)"
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 1

shared_bin_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin
cache_tool_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/bin/ccache"
cache_lib_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/lib/x86_64-linux-gnu"
install_lib_2054="$repo_root_2054/.artifacts/install-hybrid-2054-sm90/lib"
run_dir_2054="$repo_root_2054/.artifacts/benchmarks/hybrid-provider-2054/installed-48-holdout"
mkdir -p "$run_dir_2054"
export PATH="$(dirname "$cache_tool_2054"):$shared_bin_2054:$PATH"
export LD_LIBRARY_PATH="$cache_lib_2054${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root_2054/.artifacts/python-site:$repo_root_2054/python:$repo_root_2054"
export PYTHONUTF8=1
export CCACHE_DIR="$repo_root_2054/.artifacts/compiler-cache/data"
export CCACHE_BASEDIR="$repo_root_2054"
export GENERATIVEQC_LIBRARY="$install_lib_2054/libgenerativeqc.so"
python_2054="$shared_bin_2054/python"
test -x "$cache_tool_2054" && test -x "$python_2054"

verify_sha_2054() {
  local expected_2054="$1" path_2054="$2"
  test "$(sha256sum "$path_2054" | cut -d' ' -f1)" = "$expected_2054"
}
verify_sha_2054 0f8fb7fc05befa133e1dbd44b9d725cfe6789adeb0db0e75f8a70f02ada088b2 "$install_lib_2054/libgenerativeqc.so"
verify_sha_2054 49d3c8d999ad8f00a99f6f2874ecd772ed8a01b74b7da53f4f5a11cd215e9044 "$install_lib_2054/libgenerativeqc_stationary_pbe0_rks.so"
verify_sha_2054 fab751d7ad76012213ba64d56513f2693c74c6e9b3b54ac5bf88292ce548fbd0 "$install_lib_2054/libgenerativeqc_stationary_pbe0_rks_spd.so"
verify_sha_2054 3a9dbe9fc6316811e4807e1115995d817067b146317f623146f546d766d9ecbb "$install_lib_2054/generativeqc_stationary_pbe0_rks.json"
verify_sha_2054 866839a68779637ce658e5cff1cde873620016883af7fd5bb6e7eae6738b70c7 "$install_lib_2054/generativeqc_stationary_pbe0_rks_spd.json"

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'inspire_job_name=%s\nsource_head=%s\n' "$GENERATIVEQC_2054_FINITE_JOB" "$source_head_2054"
  sha256sum "$install_lib_2054"/libgenerativeqc*.so "$install_lib_2054"/generativeqc_stationary_pbe0_rks*.json
  "$cache_tool_2054" --version
  "$python_2054" --version
  "$python_2054" -m pip show numpy pyscf cupy-cuda12x | grep -E '^(Name|Version):'
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$run_dir_2054/receipt.txt" 2>&1

"$python_2054" tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-48 formaldehyde --output "$run_dir_2054/native"
for case_2054 in water-48 formaldehyde; do
  for arm_2054 in direct df-jk-occupied; do
    "$python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
      --case "$case_2054" --arm "$arm_2054" \
      --trace "$run_dir_2054/${case_2054}-${arm_2054}-trace.jsonl" \
      --output "$run_dir_2054/${case_2054}-${arm_2054}-profile.json"
  done
done

# This checks native completeness only. Independent PySCF force gates run in
# separate CPU processes before any numerical acceptance or crossover summary.
"$python_2054" - "$run_dir_2054" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "native/campaign.json").read_text())
if manifest.get("status") != "RECORDED" or len(manifest.get("order", [])) != 10:
    raise SystemExit("#2054 large ABBA manifest is incomplete")
statuses = []
for path in sorted((root / "native").glob("*.json")):
    if path.name != "campaign.json":
        statuses.append((path.name, json.loads(path.read_text()).get("status")))
for path in sorted(root.glob("*-profile.json")):
    record = json.loads(path.read_text())
    statuses.append((path.name, record.get("status")))
    if path.name.endswith("df-jk-occupied-profile.json") and not record.get("occupied_reuse_verified"):
        statuses.append((path.name + ":occupied-reuse", "FAIL"))
print(statuses, flush=True)
if len(statuses) != 14 or any(status not in {"MEASURED", "UNSUPPORTED"} for _, status in statuses):
    raise SystemExit("#2054 large native pilot incomplete; inspect all raw attempts")
print("NATIVE_RECORDED; independent PySCF oracle and full #2054 gates pending", flush=True)
PY

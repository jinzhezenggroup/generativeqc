#!/usr/bin/env bash
# Complete the exact frozen ed26480e H200 build's PBE0 stationary-force assets.
set -euo pipefail

repo_root_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/issue-2054-crossover-20261008
source_head_2054=ed26480ecccef91506b411c773d4730760a04850
core_sha_2054=2e588e90b4e6b91ea31841e940e0ceb8e543ef7dfcf110785e614abb8ec492d4
cd "$repo_root_2054"
: "${GENERATIVEQC_2054_FINITE_JOB:?finite Inspire GPU Job name required}"
test "$(git rev-parse HEAD)" = "$source_head_2054"
test -z "$(git status --porcelain)"
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 1

shared_bin_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin
cache_tool_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/bin/ccache"
cache_lib_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/lib/x86_64-linux-gnu"
build_dir_2054="$repo_root_2054/.artifacts/build-hybrid-2054-sm90-cuda129"
run_dir_2054="$repo_root_2054/.artifacts/benchmarks/hybrid-provider-2054/cuda129-aot-smoke"
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
test "$(sha256sum "$GENERATIVEQC_LIBRARY" | cut -d' ' -f1)" = "$core_sha_2054"

receipt_2054="$run_dir_2054/aot-receipt.txt"
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'inspire_job_name=%s\nsource_head=%s\ncore_sha256=%s\n' \
    "$GENERATIVEQC_2054_FINITE_JOB" "$source_head_2054" "$core_sha_2054"
  printf 'cache_launcher=%s\n' "$cache_tool_2054"
  "$cache_tool_2054" --version
  "$cache_tool_2054" --show-stats
  nvcc --version
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$receipt_2054" 2>&1

# The public PBE0/def2-SVP force route is fail-closed for pbe0_rks_spd.
# Build both declared PBE0 RKS component domains and their sealed manifests.
if ! cmake --build "$build_dir_2054" \
  --target generativeqc_stationary_pbe0_rks_manifest \
           generativeqc_stationary_pbe0_rks_spd_manifest \
  --parallel 6 > "$run_dir_2054/aot-build.log" 2>&1; then
  tail -n 100 "$run_dir_2054/aot-build.log" >&2
  exit 1
fi
for component_2054 in pbe0_rks pbe0_rks_spd; do
  manifest_2054="$build_dir_2054/generativeqc_stationary_${component_2054}.json"
  library_2054="$build_dir_2054/libgenerativeqc_stationary_${component_2054}.so"
  test -f "$manifest_2054" && test -f "$library_2054"
  sha256sum "$manifest_2054" "$library_2054" >> "$receipt_2054"
done
"$cache_tool_2054" --show-stats >> "$receipt_2054"
test "$(sha256sum "$GENERATIVEQC_LIBRARY" | cut -d' ' -f1)" = "$core_sha_2054"

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
    raise SystemExit("#2054 AOT small device pilot failed; inspect retained raw records")
PY

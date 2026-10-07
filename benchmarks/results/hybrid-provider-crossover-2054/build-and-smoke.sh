#!/usr/bin/env bash
# Source-matched sm_90 portable build and three-atom pilot in one finite GPU Job.
set -euo pipefail

repo_root_2054="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root_2054"
: "${SLURM_JOB_ID:?finite GPU Job required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm device visibility required}"
test -z "$(git status --porcelain)" || {
  echo "refusing dirty #2054 source checkout" >&2
  exit 1
}

shared_python_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin/python
shared_bin_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin
cache_tool_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/bin/ccache"
cache_lib_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/lib/x86_64-linux-gnu"
test -x "$shared_python_2054" && test -x "$cache_tool_2054" && test -d "$cache_lib_2054"
export PATH="$shared_bin_2054:$PATH"
export LD_LIBRARY_PATH="$cache_lib_2054${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root_2054/.artifacts/python-site:$repo_root_2054/python:$repo_root_2054"
export PYTHONUTF8=1
export CCACHE_DIR="$repo_root_2054/.artifacts/compiler-cache/data"
export CCACHE_BASEDIR="$repo_root_2054"

run_root_2054="$repo_root_2054/.artifacts/benchmarks/hybrid-provider-2054"
build_dir_2054="$repo_root_2054/.artifacts/build-hybrid-2054-sm90"
mkdir -p "$run_root_2054" "$CCACHE_DIR"
receipt_2054="$run_root_2054/build-receipt.txt"
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'slurm_job_id=%s\nsource_head=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)"
  git status --porcelain
  sha256sum tools/benchmark_hybrid_provider_crossover.py python/generativeqc/calculator.py src/dft/cuda_ks.cpp
  "$shared_python_2054" --version
  "$shared_python_2054" -m pip show numpy pyscf cupy-cuda12x | grep -E '^(Name|Version):'
  nvcc --version
  cmake --version
  ninja --version
  "$cache_tool_2054" --version
  "$cache_tool_2054" --show-stats
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$receipt_2054" 2>&1

# H200 has no measured generated-shell profile; freeze an explicit portable
# sm_90 build. This is a device pilot, never evidence for a tuned default.
if ! cmake -S . -B "$build_dir_2054" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=90 \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=90 \
  -DGENERATIVEQC_AOT_PROFILE=portable \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_tool_2054" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache_tool_2054" \
  -DPython_EXECUTABLE="$shared_python_2054" \
  > "$run_root_2054/configure.log" 2>&1; then
  tail -n 100 "$run_root_2054/configure.log" >&2
  exit 1
fi
grep -E '^(CMAKE_CXX_COMPILER_LAUNCHER|CMAKE_CUDA_COMPILER_LAUNCHER|GENERATIVEQC_AOT_PROFILE|GENERATIVEQC_CUDA_ARCHITECTURES):' \
  "$build_dir_2054/CMakeCache.txt" >> "$receipt_2054"
if ! cmake --build "$build_dir_2054" --target generativeqc --parallel 6 \
  > "$run_root_2054/build.log" 2>&1; then
  tail -n 100 "$run_root_2054/build.log" >&2
  exit 1
fi
"$cache_tool_2054" --show-stats >> "$receipt_2054"
export GENERATIVEQC_LIBRARY="$build_dir_2054/libgenerativeqc.so"
test -f "$GENERATIVEQC_LIBRARY"
sha256sum "$GENERATIVEQC_LIBRARY" >> "$receipt_2054"

"$shared_python_2054" -m pytest tests/python/test_hybrid_provider_crossover.py \
  -q --basetemp "$run_root_2054/pytest-smoke" \
  > "$run_root_2054/pytest-smoke.txt" 2>&1
"$shared_python_2054" tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-3 --output "$run_root_2054/native-smoke"
"$shared_python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
  --case water-3 --arm direct \
  --trace "$run_root_2054/direct-smoke-trace.jsonl" \
  --output "$run_root_2054/direct-smoke-profile.json"
"$shared_python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
  --case water-3 --arm df-jk-occupied \
  --trace "$run_root_2054/df-smoke-trace.jsonl" \
  --output "$run_root_2054/df-smoke-profile.json"

"$shared_python_2054" - "$run_root_2054" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "native-smoke/campaign.json").read_text())
if manifest.get("status") != "RECORDED" or len(manifest.get("order", [])) != 5:
    raise SystemExit("#2054 ABBA smoke manifest is incomplete")
records = sorted(
    path for path in (root / "native-smoke").glob("*.json")
    if path.name != "campaign.json"
)
statuses = [(path.name, json.loads(path.read_text()).get("status")) for path in records]
for path in (root / "direct-smoke-profile.json", root / "df-smoke-profile.json"):
    value = json.loads(path.read_text())
    statuses.append((path.name, value.get("status")))
    if path.name.startswith("df-") and not value.get("occupied_reuse_verified"):
        statuses.append(("occupied-reuse", "FAIL"))
print(statuses, flush=True)
if not statuses or any(status not in {"MEASURED", "UNSUPPORTED"} for _, status in statuses):
    raise SystemExit("#2054 small device pilot failed; inspect retained raw records")
PY

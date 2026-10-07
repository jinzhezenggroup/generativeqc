#!/usr/bin/env bash
# Build and install one source-matched H200 PBE0 native runtime before timing.
set -euo pipefail

repo_root_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/issue-2054-crossover-20261008
cd "$repo_root_2054"
: "${GENERATIVEQC_2054_FINITE_JOB:?finite Inspire GPU Job name required}"
test -z "$(git status --porcelain)"
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 1

shared_bin_2054=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin
cache_tool_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/bin/ccache"
cache_lib_2054="$repo_root_2054/.artifacts/compiler-cache/extracted/usr/lib/x86_64-linux-gnu"
build_dir_2054="$repo_root_2054/.artifacts/build-hybrid-2054-installed-sm90"
install_dir_2054="$repo_root_2054/.artifacts/install-hybrid-2054-sm90"
run_dir_2054="$repo_root_2054/.artifacts/benchmarks/hybrid-provider-2054/installed-smoke"
mkdir -p "$run_dir_2054"
test -x "$shared_bin_2054/python" && test -x "$cache_tool_2054"
export PATH="$(dirname "$cache_tool_2054"):$shared_bin_2054:$PATH"
export LD_LIBRARY_PATH="$cache_lib_2054${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root_2054/.artifacts/python-site:$repo_root_2054/python:$repo_root_2054"
export PYTHONUTF8=1
export CCACHE_DIR="$repo_root_2054/.artifacts/compiler-cache/data"
export CCACHE_BASEDIR="$repo_root_2054"
python_2054="$shared_bin_2054/python"
receipt_2054="$run_dir_2054/build-install-receipt.txt"

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'inspire_job_name=%s\nsource_head=%s\n' \
    "$GENERATIVEQC_2054_FINITE_JOB" "$(git rev-parse HEAD)"
  sha256sum tools/benchmark_hybrid_provider_crossover.py python/generativeqc/calculator.py src/dft/cuda_ks.cpp CMakeLists.txt
  "$python_2054" --version
  "$python_2054" -m pip show numpy pyscf cupy-cuda12x | grep -E '^(Name|Version):'
  nvcc --version
  cmake --version
  ninja --version
  "$cache_tool_2054" --version
  "$cache_tool_2054" --show-stats
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$receipt_2054" 2>&1

if ! cmake -S . -B "$build_dir_2054" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=90 \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=90 \
  -DGENERATIVEQC_AOT_PROFILE=portable \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF \
  -DGENERATIVEQC_STATIONARY_AOT_PROFILES=pbe0_rks \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_tool_2054" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$cache_tool_2054" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DPython_EXECUTABLE="$python_2054" \
  > "$run_dir_2054/configure.log" 2>&1; then
  tail -n 100 "$run_dir_2054/configure.log" >&2
  exit 1
fi
grep -E '^(CMAKE_CXX_COMPILER_LAUNCHER|CMAKE_CUDA_COMPILER_LAUNCHER|GENERATIVEQC_STATIONARY_AOT_PROFILES|GENERATIVEQC_CUDA_ARCHITECTURES):' \
  "$build_dir_2054/CMakeCache.txt" >> "$receipt_2054"
ninja -C "$build_dir_2054" -t commands generativeqc \
  > "$run_dir_2054/compile-commands.txt"
"$python_2054" - "$run_dir_2054/compile-commands.txt" "$cache_tool_2054" \
  >> "$receipt_2054" <<'PY'
import sys
from pathlib import Path

commands = Path(sys.argv[1]).read_text().splitlines()
cache = sys.argv[2]
for compiler in ("nvcc", "c++"):
    matches = [line for line in commands if cache in line and compiler in line]
    if not matches:
        raise SystemExit(f"missing actual cached {compiler} compile command")
    print(f"cached_{compiler}_command={matches[0][:500]}")
PY
if ! cmake --build "$build_dir_2054" --parallel 6 \
  > "$run_dir_2054/build.log" 2>&1; then
  tail -n 100 "$run_dir_2054/build.log" >&2
  exit 1
fi
if ! cmake --install "$build_dir_2054" --prefix "$install_dir_2054" \
  > "$run_dir_2054/install.log" 2>&1; then
  tail -n 100 "$run_dir_2054/install.log" >&2
  exit 1
fi
"$cache_tool_2054" --show-stats >> "$receipt_2054"

export GENERATIVEQC_LIBRARY="$install_dir_2054/lib/libgenerativeqc.so"
for name_2054 in libgenerativeqc.so \
  libgenerativeqc_stationary_pbe0_rks.so \
  libgenerativeqc_stationary_pbe0_rks_spd.so \
  generativeqc_stationary_pbe0_rks.json \
  generativeqc_stationary_pbe0_rks_spd.json; do
  test -f "$install_dir_2054/lib/$name_2054"
  sha256sum "$install_dir_2054/lib/$name_2054" >> "$receipt_2054"
done

"$python_2054" -m pytest tests/python/test_hybrid_provider_crossover.py \
  -q --basetemp "$run_dir_2054/pytest" > "$run_dir_2054/pytest.txt" 2>&1
"$python_2054" tools/benchmark_hybrid_provider_crossover.py run-campaign \
  --cases water-3 --output "$run_dir_2054/native"
for arm_2054 in direct df-jk-occupied; do
  "$python_2054" tools/benchmark_hybrid_provider_crossover.py run-profile \
    --case water-3 --arm "$arm_2054" \
    --trace "$run_dir_2054/${arm_2054}-trace.jsonl" \
    --output "$run_dir_2054/${arm_2054}-profile.json"
  "$python_2054" tools/benchmark_hybrid_provider_crossover.py run-reference \
    --native "$run_dir_2054/native/water-3-${arm_2054}-0.json" \
    --threads 8 --output "$run_dir_2054/oracle-${arm_2054}-0.json"
done
"$python_2054" tools/benchmark_hybrid_provider_crossover.py summarize \
  "$run_dir_2054/native/water-3-direct-0.json" \
  "$run_dir_2054/native/water-3-df-jk-occupied-0.json" \
  "$run_dir_2054/native/water-3-df-j-exact-k.json" \
  --oracle "$run_dir_2054/oracle-direct-0.json" \
  --oracle "$run_dir_2054/oracle-df-jk-occupied-0.json" \
  --profile "$run_dir_2054/direct-profile.json" \
  --profile "$run_dir_2054/df-jk-occupied-profile.json" \
  --output "$run_dir_2054/summary-pair0.json"
"$python_2054" - "$run_dir_2054/summary-pair0.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
print({"pilot_status": record.get("status"), "failures": record.get("failures")}, flush=True)
if record.get("status") != "PILOT_ACCEPTED" or record.get("crossover_claim_eligible"):
    raise SystemExit("installed #2054 small pilot did not meet its independent gate")
PY

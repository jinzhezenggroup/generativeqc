#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_HEAD:?pin the reviewed source commit}"
: "${INSPIRE_JOB_NAME:?record the scheduler job name}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-2072-point-family
run="$root/runs/$INSPIRE_JOB_NAME"
venv="$root/venv"
build="$repo/build/issue-2072-sm90-cuda129-shell-aot"
mkdir -p "$run"
cd "$repo"
export CUDA_PATH="$root/cuda-12.9"
export PATH="$root/tools/bin:$venv/bin:$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$root/tools/lib:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test "$(sha256sum "$CUDA_PATH/bin/nvcc" | cut -d ' ' -f 1)" = \
  df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"

export PYTHONPATH="$repo/python:$repo"
export GENERATIVEQC_LIBRARY="$build/libgenerativeqc.so"
export GENERATIVEQC_STATIONARY_CACHE="$run/jit-cache"
export SCCACHE_DIR="$root/sccache"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1

if [ "${1:-}" = --probe ]; then
  git --version
  cmake --version
  ninja --version
  g++ --version
  nvcc --version
  "$venv/bin/sccache" --version
  "$venv/bin/python" -c 'import cupy as cp, numpy, scipy, pyscf; print("CUDA devices:", cp.cuda.runtime.getDeviceCount())'
  exit 0
fi

"$venv/bin/python" - "$repo/python/generativeqc_compiler/integral/production_shell_classes.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    profile = json.load(source)["architectures"]["portable_cuda"]
if not profile["kernels"]:
    raise SystemExit(
        "H100 complete E+F qualification requires a qualified shell force profile; "
        "portable_cuda currently has no production shell kernels"
    )
PY

"$venv/bin/sccache" --version | tee "$run/sccache-version.txt"
nvcc --version > "$run/nvcc-version.txt"
sha256sum "$CUDA_PATH/bin/nvcc" > "$run/nvcc.sha256"
nvidia-smi -L > "$run/device.txt"
git rev-parse HEAD > "$run/source-head.txt"
"$venv/bin/sccache" --show-stats > "$run/sccache-before.txt"
trap 'status=$?; "$venv/bin/sccache" --show-stats > "$run/sccache-after.txt"; printf "%s\n" "$status" > "$run/exit.txt"' EXIT

cmake -S "$repo" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$venv/bin/sccache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$venv/bin/sccache" \
  -DPython3_EXECUTABLE="$venv/bin/python" \
  -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=90-real \
  -DGENERATIVEQC_AOT_PROFILE=portable \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  -DGENERATIVEQC_STATIONARY_AOT_PROFILES=pbe0_rks \
  -DGENERATIVEQC_AOT_UNIT_MODE=stable-shards \
  -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
  -DGENERATIVEQC_CUDA_SEPARABLE_COMPILATION=OFF \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=2 \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
  -DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=2 \
  -DGENERATIVEQC_BUILD_TESTS=ON \
  -DGENERATIVEQC_BUILD_CLI=OFF > "$run/configure.log" 2>&1 || {
    tail -n 100 "$run/configure.log"; exit 1;
  }
cmake --build "$build" --parallel 8 --target generativeqc generativeqc_dft_cuda_tests \
  generativeqc_stationary_pbe0_rks_manifest \
  generativeqc_stationary_pbe0_rks_spd_manifest \
  > "$run/build.log" 2>&1 || { tail -n 100 "$run/build.log"; exit 1; }
grep -m 3 sccache "$build/build.ninja" > "$run/launcher-check.txt"
sha256sum "$build/libgenerativeqc.so" \
  "$build/generativeqc_stationary_pbe0_rks.json" \
  "$build/libgenerativeqc_stationary_pbe0_rks.so" \
  "$build/generativeqc_stationary_pbe0_rks_spd.json" \
  "$build/libgenerativeqc_stationary_pbe0_rks_spd.so" > "$run/artifacts.sha256"

for mode in 0 1; do
  export GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION="$mode"
  "$build/generativeqc_dft_cuda_tests" > "$run/native-$mode.log" 2>&1
  "$build/generativeqc_dft_cuda_tests" --point-batches > "$run/point-batches-$mode.log" 2>&1
  "$build/generativeqc_dft_cuda_tests" --pbe0-local-ao > "$run/pbe0-local-ao-$mode.log" 2>&1
done

gzip -dc benchmarks/results/pbe0-xc-tiles-20261006/reference-48.json.gz \
  > "$run/reference-48.json"
for atoms in 48 96; do
  if [ "$atoms" = 96 ]; then
    gzip -dc benchmarks/results/pbe0-xc-tiles-20261006/reference-96.json.gz \
      > "$run/reference-96.json"
  fi
  "$venv/bin/python" -m benchmarks.pbe0_xc_tile_pairs \
    --atoms "$atoms" \
    --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
    --reference "$run/reference-$atoms.json" \
    --point-specialization --feasibility --repeats 1 \
    --output "$run/feasibility-$atoms.json" > "$run/feasibility-$atoms.log" 2>&1
done

for atoms in 48 96; do
  "$venv/bin/python" -m benchmarks.pbe0_xc_tile_pairs \
    --atoms "$atoms" \
    --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
    --reference "$run/reference-$atoms.json" \
    --point-specialization --repeats 5 \
    --output "$run/pairs-$atoms.json" > "$run/pairs-$atoms.log" 2>&1

  "$venv/bin/python" - "$run" "$atoms" > "$run/audit-$atoms.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from tools.generativeqc_validation.pbe0_xc_tiles import verify_point_family_pairs

run = Path(sys.argv[1])
atoms = int(sys.argv[2])
record = json.loads((run / f"pairs-{atoms}.json").read_text())
verify_point_family_pairs(
    record,
    json.loads((run / f"reference-{atoms}.json").read_text()),
)
print(f"{atoms}-atom fixed-work PBE0 point-family evidence: PASS")
for phase, assessment in record["assessments"].items():
    print(f"{phase} performance: {assessment['status']}; {assessment.get('reason')}")
PY
done

#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
: "${CUDA_ROOT:?set CUDA_ROOT to the actual CUDA 12.9 toolkit}"
export CUDA_PATH="$CUDA_ROOT" PATH="$CUDA_ROOT/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
out="$PWD/.artifacts/becke-physical-forces-reproduction"
mkdir -p "$out/bin" "$out/compiler-tmp"
export TMPDIR="$out/compiler-tmp" CCACHE_BASEDIR="$PWD"
gzip -cd "$bundle/source-files.sha256.gz" | sha256sum --check > "$out/source-check.txt"
ccache --version > "$out/ccache-version.txt"
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    ccache --show-stats > "$out/core-ccache-before.txt"
    export CCACHE_LOGFILE="$out/core-ccache.log"
    cmake --preset cuda-release-sm120 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DPython3_EXECUTABLE="$(command -v "$python")" \
        -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v ccache)" \
        -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v ccache)" \
        -DGENERATIVEQC_CUDA_COMPILE_JOBS=2 -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
        -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF \
        -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF \
        -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF
    ninja -C build/cuda-release-sm120 -t commands generativeqc > "$out/core-commands.txt"
    "$python" - "$out/core-commands.txt" <<'PY'
import sys
from pathlib import Path
commands = [line for line in Path(sys.argv[1]).read_text().splitlines() if ' -c ' in line and ('nvcc ' in line or 'g++ ' in line or 'c++ ' in line)]
assert commands and all('ccache ' in line for line in commands)
print('Verified cached compiler commands:', len(commands))
PY
    cmake --build build/cuda-release-sm120 --target generativeqc --parallel 4
    ccache --show-stats > "$out/core-ccache-after.txt"
    exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --time=00:25:00 bash "$bundle/reproduce.sh"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve assigned scheduler visibility}"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
test -f "$GENERATIVEQC_LIBRARY"
sha256sum "$GENERATIVEQC_LIBRARY" > "$out/core.sha256"
printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$(command -v ccache)" "$CUDA_ROOT/bin/nvcc" > "$out/bin/nvcc"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$CUDA_ROOT/bin/ptxas" > "$out/bin/ptxas"
chmod +x "$out/bin/nvcc" "$out/bin/ptxas"
export GENERATIVEQC_NVCC="$out/bin/nvcc" CUDACXX="$out/bin/nvcc"
export GENERATIVEQC_STATIONARY_CACHE="$out/cache" CCACHE_LOGFILE="$out/ccache.log"
export LD_LIBRARY_PATH="$PWD/build/cuda-release-sm120:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CUDA_ROOT/lib64/libcudart.so.12:$CUDA_ROOT/lib64/libcublas.so.12:$CUDA_ROOT/lib64/libcublasLt.so.12:$CUDA_ROOT/lib64/libcusolver.so.11:$CUDA_ROOT/lib64/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export GENERATIVEQC_TEST_WB97MV_CUDA=1 GENERATIVEQC_WB97MV_BECKE_PRIMITIVE_TEST=1
export GENERATIVEQC_WB97MV_BECKE_EVIDENCE="$out/physical-forces"
trap 'status=$?; ccache --show-stats > "$out/ccache-after.txt"; printf "%s\n" "$status" > "$out/job.exit"; exit "$status"' EXIT
ccache --show-stats > "$out/ccache-before.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$out/job.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$out/device-visibility.txt"
"$python" -m pytest -q tests/python/test_wb97mv_complete_cuda.py \
    -k 'test_complete_cuda_force_matches_independent_engine and sto-3g' \
    --basetemp="$out/pytest-nonlocal"

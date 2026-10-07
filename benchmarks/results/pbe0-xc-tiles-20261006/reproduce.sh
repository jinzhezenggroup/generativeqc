#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${CUDA_ROOT:?set CUDA_ROOT to the CUDA 12.9 toolkit}"
: "${PYTHON:?set PYTHON to the matching benchmark Python}"
cmake=${CMAKE:-cmake}
ccache=${CCACHE:-ccache}
host_cxx=${HOST_CXX:-/usr/bin/g++-11}
export CUDA_PATH="$CUDA_ROOT" PATH="$CUDA_ROOT/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python"
export CCACHE_BASEDIR="$PWD" CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
out="$PWD/.artifacts/xc-tile-reproduction"
mkdir -p "$out/bin"
sha256sum --check "$bundle/source-files.sha256" > "$out/source-check.txt"
"$ccache" --version > "$out/ccache-version.txt"
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    "$ccache" --show-stats > "$out/core-ccache-before.txt"
    "$cmake" --preset cuda-release-sm120 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$host_cxx" -DCMAKE_CXX_COMPILER="$host_cxx" \
        -DPython3_EXECUTABLE="$PYTHON" -DCMAKE_CXX_COMPILER_LAUNCHER="$ccache" \
        -DCMAKE_CUDA_COMPILER_LAUNCHER="$ccache" \
        -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
        -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 -DGENERATIVEQC_AOT_COMPILE_JOBS=4 \
        -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF
    /usr/bin/time -f '%e' -o "$out/core-build-seconds.txt" \
        "$cmake" --build build/cuda-release-sm120 --parallel 8 --target generativeqc
    "$ccache" --show-stats > "$out/core-ccache-after.txt"
    exec srun --partition=main --gres="${SLURM_GRES:-gpu:pro6000:1}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --time="${SLURM_TIME:-00:50:00}" \
        bash "$bundle/reproduce.sh"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$ccache" "$CUDA_ROOT/bin/nvcc" > "$out/bin/nvcc"
printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$ccache" "$host_cxx" > "$out/bin/c++"
chmod +x "$out/bin/nvcc" "$out/bin/c++"
export CUDACXX="$out/bin/nvcc" GENERATIVEQC_NVCC="$out/bin/nvcc"
export NVCC="$out/bin/nvcc" NVCC_CCBIN="$host_cxx" CXX="$out/bin/c++"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CUDA_ROOT/lib64/libcudart.so.12:$CUDA_ROOT/lib64/libcublas.so.12:$CUDA_ROOT/lib64/libcublasLt.so.12:$CUDA_ROOT/lib64/libcusolver.so.11:$CUDA_ROOT/lib64/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export GENERATIVEQC_STATIONARY_CACHE="$out/cache"
export GENERATIVEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS=8
trap 'status=$?; "$ccache" --show-stats > "$out/ccache-after.txt"; printf "%s\n" "$status" > "$out/job.exit"' EXIT
"$ccache" --show-stats > "$out/ccache-before.txt"
"$PYTHON" - <<'PY'
import ctypes
from pathlib import Path
from generativeqc.autotune import source_identity
library = ctypes.CDLL(str(Path('build/cuda-release-sm120/libgenerativeqc.so').resolve()))
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == source_identity(Path.cwd())
assert source_identity(Path.cwd()) == 'b0e32abf342aa53877f2a350801a1dfdc1bc74d341df0cebe82a738d74429d2f'
PY
basis=benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json
for atoms in 96 48; do
    if [[ ${FRESH_REFERENCE:-0} == 1 ]]; then
        "$PYTHON" -m benchmarks.readme_pbe0 reference --atoms "$atoms" \
            --basis-file "$basis" --repeats 5 --output "$out/reference-$atoms.json"
    else
        gzip -dc "$bundle/reference-$atoms.json.gz" > "$out/reference-$atoms.json"
    fi
    "$PYTHON" -m benchmarks.pbe0_xc_tile_pairs --atoms "$atoms" --basis-file "$basis" \
        --reference "$out/reference-$atoms.json" --repeats 5 --output "$out/pairs-$atoms.json"
done

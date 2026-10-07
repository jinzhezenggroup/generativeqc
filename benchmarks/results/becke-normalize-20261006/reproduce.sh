#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${CUDA_ROOT:?set CUDA_ROOT to the actual CUDA 12.9 toolkit}"
python=${PYTHON:-python3}
host_cxx=${HOST_CXX:-/usr/bin/g++-11}
export CUDA_PATH="$CUDA_ROOT" PATH="$CUDA_ROOT/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python"
export CCACHE_BASEDIR="$PWD" CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
out="$PWD/.artifacts/becke-normalize-reproduction"
mkdir -p "$out/bin"
sha256sum --check "$bundle/source-files.sha256" > "$out/source-check.txt"
ccache --version > "$out/ccache-version.txt"
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    ccache --show-stats > "$out/core-ccache-before.txt"
    cmake --preset cuda-release-sm120 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$host_cxx" -DCMAKE_CXX_COMPILER="$host_cxx" \
        -DPython3_EXECUTABLE="$(command -v "$python")" \
        -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v ccache)" \
        -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v ccache)" \
        -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
        -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF \
        -DGENERATIVEQC_CUDA_COMPILE_JOBS=2 -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
        -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF
    /usr/bin/time -f '%e' -o "$out/core-compile-seconds.txt" \
        cmake --build build/cuda-release-sm120 --parallel 8 --target generativeqc \
        generativeqc_stationary_pbe_rks.json generativeqc_stationary_pbe_uks.json
    ccache --show-stats > "$out/core-ccache-after.txt"
    exec srun --partition=main --gres="${SLURM_GRES:-gpu:pro6000:1}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 bash "$bundle/reproduce.sh"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$(command -v ccache)" "$CUDA_ROOT/bin/nvcc" > "$out/bin/nvcc"
printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$(command -v ccache)" "$host_cxx" > "$out/bin/c++"
chmod +x "$out/bin/nvcc" "$out/bin/c++"
export NVCC="$out/bin/nvcc" NVCC_CCBIN="$host_cxx"
export CUDACXX="$out/bin/nvcc" GENERATIVEQC_NVCC="$out/bin/nvcc" CXX="$out/bin/c++"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CUDA_ROOT/lib64/libcudart.so.12:$CUDA_ROOT/lib64/libcublas.so.12:$CUDA_ROOT/lib64/libcublasLt.so.12:$CUDA_ROOT/lib64/libcusolver.so.11:$CUDA_ROOT/lib64/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export GENERATIVEQC_STATIONARY_CACHE="$out/cache"
export GENERATIVEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS=8
export GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE=off
scontrol show job "$SLURM_JOB_ID" -o > "$out/job.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$out/visibility.txt"
trap 'status=$?; ccache --show-stats > "$out/ccache-after.txt"; printf "%s\n" "$status" > "$out/job.exit"' EXIT
ccache --show-stats > "$out/ccache-before.txt"
"$python" - <<'PY'
import ctypes
from pathlib import Path
from generativeqc.autotune import source_identity
library = ctypes.CDLL(str(Path('build/cuda-release-sm120/libgenerativeqc.so').resolve()))
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == source_identity(Path.cwd())
PY
basis=benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json
for atoms in 96 48; do
    "$python" -m benchmarks.readme_pbe0 reference --atoms "$atoms" \
        --basis-file "$basis" --repeats 5 --output "$out/reference-$atoms.json"
    "$python" -m benchmarks.becke_normalize_pairs --atoms "$atoms" \
        --basis-file "$basis" --reference "$out/reference-$atoms.json" \
        --repeats 5 --output "$out/pairs-$atoms.json"
done
export GENERATIVEQC_STATIONARY_EVIDENCE="$out/physical"
"$python" -m pytest -q -s tests/python/test_becke_normalize_cooperative.py
unset GENERATIVEQC_STATIONARY_EVIDENCE
for sanitizer in memcheck racecheck initcheck synccheck; do
    timeout 600 compute-sanitizer --tool "$sanitizer" --error-exitcode 87 \
        "$python" -m pytest -q tests/python/test_becke_normalize_cooperative.py \
        -k cuda_normalize --basetemp="$out/pytest-$sanitizer" > "$out/$sanitizer.log" 2>&1
done
"$python" -m pytest -q tests/python/test_dft_stationary_native.py
"$python" tools/generate_stationary_force_aot.py --profile pbe0_rks --output "$out/owner.cu"
/usr/bin/time -f '%e' -o "$out/wrapper-compile-seconds.txt" \
    "$out/bin/nvcc" -O3 -std=c++20 --expt-relaxed-constexpr --fmad=false \
    --split-compile=8 -arch=sm_120 -Isrc -Xcompiler=-fPIC -c "$out/owner.cu" -o "$out/owner.o"
"$CUDA_ROOT/bin/nvcc" -shared "$out/owner.o" -lcublas -o "$out/owner.so"
export GENERATIVEQC_PHASED_OWNER_PROBE="$out/owner.so"
"$python" -m pytest -q tests/python/test_stationary_becke_phased_cuda.py
if [[ ${PROFILE:-0} == 1 ]]; then
    "$python" -m benchmarks.becke_normalize_pairs --atoms 96 --basis-file "$basis" \
        --reference "$out/reference-96.json" --repeats 1 --feasibility --profile \
        --output "$out/profile-96.json"
fi

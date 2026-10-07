#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=00:25:00 bash "$bundle/reproduce.sh"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
: "${CUDA_ROOT:?set actual CUDA12.9 toolkit}"
python=${PYTHON:-python3}
out="$PWD/.artifacts/becke-native-cap-reproduction"
mkdir -p "$out/compiler-tmp"
export TMPDIR="$out/compiler-tmp" CCACHE_BASEDIR="$PWD"
export PATH="$CUDA_ROOT/bin:$PATH" CUDA_PATH="$CUDA_ROOT"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
gzip -cd "$bundle/source-files.sha256.gz" | sha256sum --check > "$out/source-check.txt"
ccache --version > "$out/ccache-version.txt"
ccache --show-stats > "$out/ccache-before.txt"
trap 'status=$?; ccache --show-stats > "$out/ccache-after.txt"; printf "%s\n" "$status" > "$out/job.exit"; exit "$status"' EXIT
"$python" tools/generate_stationary_force_aot.py --profile pbe0_rks --output "$out/owner.cu"
ccache nvcc -O3 -std=c++20 --expt-relaxed-constexpr --fmad=false -arch=sm_120 -Isrc \
    -Xcompiler=-fPIC -c "$out/owner.cu" -o "$out/owner.o"
nvcc -shared "$out/owner.o" -lcublas -o "$out/owner.so"
sha256sum "$out/owner.cu" "$out/owner.o" "$out/owner.so" > "$out/compiled.sha256"
export GENERATIVEQC_PHASED_OWNER_PROBE="$out/owner.so" GENERATIVEQC_BECKE_PHASE_RECORD="$out/phase-records.jsonl"
"$python" -m pytest -q tests/python/test_stationary_becke_phased_cuda.py -k 128 --basetemp="$out/pytest-cap"
unset GENERATIVEQC_BECKE_PHASE_RECORD
for sanitizer in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --tool "$sanitizer" --error-exitcode 87 "$python" -m pytest -q \
        'tests/python/test_stationary_becke_phased_cuda.py::test_shared_owner_phases_preserve_sources_and_work[True-True-True-full-False-128]' \
        --basetemp="$out/pytest-$sanitizer"
done

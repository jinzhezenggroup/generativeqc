#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --time=00:25:00 bash "$bundle/reproduce.sh"
fi
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm visibility}"
python=${PYTHON:-python}
if [[ -n ${CUDA_ROOT:-} ]]; then
    export PATH="$CUDA_ROOT/bin:$PATH" CUDA_PATH="$CUDA_ROOT"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CCACHE_BASEDIR="$PWD" PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
out="$PWD/.artifacts/becke-native-phases-reproduction/job-$SLURM_JOB_ID"
mkdir -p "$out"
trap 'status=$?; ccache --show-stats > "$out/ccache-after.txt"; printf "%s\n" "$status" > "$out/job.exit"; exit "$status"' EXIT
sha256sum --check "$bundle/source-files.sha256" > "$out/source-check.txt"
ccache --version > "$out/ccache-version.txt"
ccache --show-stats > "$out/ccache-before.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$out/job.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$out/device-visibility.txt"
"$python" tools/generate_stationary_force_aot.py --profile pbe0_rks --output "$out/owner.cu"
printf '%s  %s\n' c8b9eaee3a81b552d16be7326bdc8f40f2178bf577c78008e8d8c0c12d898947 "$out/owner.cu" | sha256sum --check
ccache nvcc -O3 -std=c++20 --expt-relaxed-constexpr --fmad=false -arch=sm_120 -Isrc -Xcompiler=-fPIC -c "$out/owner.cu" -o "$out/owner.o"
nvcc -shared "$out/owner.o" -lcublas -o "$out/owner.so"
sha256sum "$out/owner.cu" "$out/owner.o" "$out/owner.so" > "$out/compiled.sha256"
export GENERATIVEQC_PHASED_OWNER_PROBE="$out/owner.so" GENERATIVEQC_BECKE_PHASE_RECORD="$out/phase-records.jsonl"
"$python" -m pytest -q tests/python/test_stationary_becke_phased_cuda.py --basetemp="$out/pytest-owner"
unset GENERATIVEQC_BECKE_PHASE_RECORD
for sanitizer in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --tool "$sanitizer" --error-exitcode 87 "$python" -m pytest -q \
        'tests/python/test_stationary_becke_phased_cuda.py::test_shared_owner_phases_preserve_sources_and_work[True-True-True-full-False-96]' \
        --basetemp="$out/pytest-$sanitizer" > "$out/$sanitizer.log" 2>&1
done

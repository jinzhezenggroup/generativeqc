#!/usr/bin/env bash
set -euo pipefail

evidence=$(cd "$(dirname "$0")" && pwd)
stage=${1:?choose build, reference, clean, trace, audit or ab}
functional=${2:-m06-2x}
atoms=${3:-96}
build=${BUILD_DIRECTORY:-build-cuda}
raw=${RAW_DIRECTORY:-.artifacts/issue2474-reproduction}
python=${PYTHON:-python}
export PYTHONPATH="python:.:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$raw"

if [[ "$stage" == build ]]; then
  launcher=$(command -v sccache || command -v ccache)
  "$launcher" --version
  "$launcher" --show-stats > "$raw/cache-before.txt"
  cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$launcher" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER="$launcher" \
    -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_CUDA_ARCHITECTURES=120 \
    -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
    -DGENERATIVEQC_CUDA_COMPILE_JOBS=8 -DGENERATIVEQC_AOT_COMPILE_JOBS=4 \
    '-DGENERATIVEQC_STATIONARY_AOT_PROFILES=pbe0_rks;b3lyp_rks;m06_2x_rks'
  cmake --build "$build" --parallel 24 --target generativeqc_ks_cuda_tests \
    generativeqc_stationary_m06_2x_rks_spd_manifest \
    generativeqc_stationary_pbe0_rks_spd_manifest \
    generativeqc_stationary_b3lyp_rks_spd_manifest
  "$launcher" --show-stats > "$raw/cache-after.txt"
  exit
fi

export GENERATIVEQC_LIBRARY="$(cd "$build" && pwd)/libgenerativeqc.so"
common=(--functional "$functional" --atoms "$atoms"
  --basis-file "$evidence/inputs/def2-svp-ho.json"
  --auxiliary-file "$evidence/inputs/cc-pvdz-jkfit.json")
reference="$raw/reference-$functional-$atoms.json"
run=(srun --partition="${PARTITION:-main}" --gres="${GPU_RESOURCE:-gpu:5090:1}"
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:50:00)
case "$stage" in
  reference)
    "${run[@]}" "$python" -m benchmarks.ks_df_warm_factor reference "${common[@]}" --output "$reference" ;;
  clean|trace)
    "${run[@]}" bash -c '
      set -euo pipefail
      stage=$1; python=$2; raw=$3; reference=$4; functional=$5; atoms=$6
      shift 6
      for variant in baseline candidate; do
        options=()
        if [[ "$stage" == trace ]]; then
          options+=(--trace-directory "$raw/trace-$functional-$variant-$atoms")
          output="$raw/trace-$variant-$functional-$atoms.json"
        else
          output="$raw/current-$variant-$functional-$atoms.json"
        fi
        "$python" -m benchmarks.ks_df_warm_factor native "$@" \
          --reference "$reference" --variant "$variant" "${options[@]}" --output "$output"
      done
    ' bash "$stage" "$python" "$raw" "$reference" "$functional" "$atoms" "${common[@]}" ;;
  audit)
    "${run[@]}" "$python" -m benchmarks.ks_df_warm_factor native "${common[@]}" \
      --reference "$reference" --variant candidate --audit-first-k \
      --trace-directory "$raw/audit-$functional-$atoms" \
      --output "$raw/audit-candidate-$functional-$atoms.json" ;;
  ab)
    "${run[@]}" "$python" "$evidence/reproduce_ab.py" "${common[@]}" \
      --reference "$reference" --output "$raw/ab-$functional-$atoms.json" ;;
  *) printf 'unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac

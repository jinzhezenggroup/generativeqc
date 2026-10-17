#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_CHECKOUT:?Set to a checkout containing the measured source revision}"
: "${NATIVE_ROOT:?Set to the directory containing the measured linked CLI and library}"
: "${REFERENCE_PYTHON:?Set to the GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 13.6.0 Python}"
: "${SLURM_JOB_ID:?Run through a finite single-GPU srun allocation}"
mode=${1:?Choose pilot, followup, or dimer}
bundle=$(cd "$(dirname "$0")" && pwd)
mkdir -p .artifacts
output=$(mktemp -d "$PWD/.artifacts/wb97mv-scaling-reproduce.XXXXXX")
git -C "$SOURCE_CHECKOUT" archive 5aa703721fdce6a220838676af615d574f0fed05 | tar -x -C "$output"
case "$mode" in
  pilot) driver=pilot-driver.py.xz ;;
  followup) driver=followup-driver.py.xz ;;
  dimer) driver=final-driver.py.xz ;;
  *) echo 'Choose pilot, followup, or dimer' >&2; exit 2 ;;
esac
xz -dc "$bundle/$driver" > "$output/benchmarks/compare_wb97mv_hessian_gpu4pyscf.py"
native=$(cd "$NATIVE_ROOT" && pwd)
printf '%s  %s\n' \
  0bc81c95f0990f92802fb3ccb6b0e4e5d61ca5796e0db3046242cde1e060a873 "$native/generativeqc" \
  af24b33b71009322c2be5ca85f947f1078be4bb87fbef75dc5cba11436c8b464 "$native/libgenerativeqc.so.0.1.0" | sha256sum -c -
export CUDA_HOME=${CUDA_HOME:-/group/software/cuda-12.9.1}
export CUDA_PATH="$CUDA_HOME" CUDA_DISABLE_PTX_JIT=1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$native:$CUDA_HOME/targets/x86_64-linux/lib:${REFERENCE_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export GENERATIVEQC_LIBRARY="$native/libgenerativeqc.so.0.1.0"
export PYTHONPATH="$output/python:$output"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$output"
overall=0
run_case() {
  label=$1
  shift
  status=0
  "$REFERENCE_PYTHON" -m benchmarks.compare_wb97mv_hessian_gpu4pyscf \
    --cli "$native/generativeqc" --reference-python "$REFERENCE_PYTHON" \
    --source-commit 5aa703721fdce6a220838676af615d574f0fed05 \
    --timeout-seconds 180 --gpu-memory-limit-mib 24576 \
    --output "$output/.artifacts/$label.json" "$@" || status=$?
  printf 'CASE_EXIT %s %s\n' "$label" "$status"
  if [[ "$status" != 0 ]]; then overall=1; fi
}
case "$mode" in
  pilot)
    for molecule in formaldehyde methanol ethane; do
      run_case "pilot2-$molecule" --cases "$molecule" --repeats 0
    done
    ;;
  followup)
    run_case followup-ethane-reference --cases ethane --engines gpu4pyscf --repeats 0
    run_case followup-controls --cases h2 water --repeats 0
    run_case followup-formaldehyde --cases formaldehyde --repeats 2
    ;;
  dimer)
    run_case water-dimer --cases water-dimer --repeats 0
    ;;
esac
printf 'Retained all outputs: %s\n' "$output/.artifacts"
exit "$overall"

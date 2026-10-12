#!/usr/bin/env bash
# Diagnostic primitive qualification; complete scientific endpoint gates stay open.
set -euo pipefail
qualification_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qualification_build=${1:?usage: qualify_structured_scf_routes.sh ABSOLUTE_OUTPUT CUDA_ROOT SM_ARCH PYTHON [run|build-only|run-only]}
qualification_cuda=${2:?explicit CUDA toolkit root required}
qualification_arch=${3:?explicit SM architecture required}
qualification_python=${4:?explicit Python interpreter required}
qualification_mode=${5:-run}
[[ $qualification_mode = run || $qualification_mode = build-only || $qualification_mode = run-only ]]
[[ $qualification_build = /* && $qualification_cuda = /* && $qualification_python = /* ]]
[[ $qualification_arch =~ ^[0-9]+$ ]]
if [[ $qualification_mode != build-only ]]; then
  [[ -n ${SLURM_JOB_ID:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
    printf 'Real-device qualification must run inside a Slurm GPU allocation.\n' >&2
    exit 2
  }
  [[ -z $(git -C "$qualification_root" status --porcelain --untracked-files=no) ]] || {
    printf 'Real-device evidence requires clean, committed source.\n' >&2
    exit 2
  }
fi
qualification_head=$(git -C "$qualification_root" rev-parse HEAD)
if [[ $qualification_mode != run-only ]]; then
  [[ ! -e $qualification_build ]]
  mkdir -p -- "$qualification_build"
  qualification_cache=$(command -v ccache) || {
    printf 'A verified ccache launcher is required.\n' >&2
    exit 2
  }
  qualification_cxx=$(command -v g++)
  qualification_nvcc="$qualification_cuda/bin/nvcc"
  "$qualification_cache" --version > "$qualification_build/cache-version.txt"
  "$qualification_cache" --show-stats > "$qualification_build/cache-before.txt"
  "$qualification_cxx" --version > "$qualification_build/cxx-version.txt"
  "$qualification_nvcc" --version > "$qualification_build/nvcc-version.txt"
  export CCACHE_BASEDIR="$qualification_root"
  exec 9> "$qualification_build/commands.log"
  export BASH_XTRACEFD=9
  set -x
  "$qualification_python" "$qualification_root/tools/generate_matrix_product_cuda.py" \
    --output "$qualification_build/generated_matrix_product_lowering.hpp"
  "$qualification_python" "$qualification_root/tools/generate_scf_density_cuda.py" \
    --output "$qualification_build/generated_scf_density_cuda.cuh"
  qualification_includes=(-I"$qualification_build" -I"$qualification_root/src" \
    -I"$qualification_root/include" -I"$qualification_cuda/include")
  qualification_objects=()
  for qualification_source in src/scf/cuda/matrix_library.cpp src/scf/cuda/runtime_support.cpp \
    src/tensor/cuda_matrix_product.cpp; do
    qualification_unit=$(basename -- "$qualification_source" .cpp)
    "$qualification_cache" "$qualification_cxx" -std=c++20 -O3 -fPIC -DGENERATIVEQC_TEST_HOOKS=1 \
      -MD -MF "$qualification_build/$qualification_unit.d" "${qualification_includes[@]}" \
      -c "$qualification_root/$qualification_source" -o "$qualification_build/$qualification_unit.o"
    qualification_objects+=("$qualification_build/$qualification_unit.o")
  done
  for qualification_source in src/scf/cuda/scf_matrix_kernels.cu \
    src/scf/cuda/scf_density_kernels.cu src/scf/cuda/basis_transform_kernels.cu \
    src/scf/cuda/structured_matrix.cu tests/native/test_structured_scf_routes.cu; do
    qualification_unit=$(basename -- "$qualification_source" .cu)
    "$qualification_cache" "$qualification_nvcc" -std=c++20 -O3 -arch="sm_$qualification_arch" \
      -ccbin "$qualification_cxx" -DGENERATIVEQC_TEST_HOOKS=1 -MD \
      -MF "$qualification_build/$qualification_unit.d" "${qualification_includes[@]}" \
      -c "$qualification_root/$qualification_source" -o "$qualification_build/$qualification_unit.o"
    qualification_objects+=("$qualification_build/$qualification_unit.o")
  done
  "$qualification_nvcc" -ccbin "$qualification_cxx" "${qualification_objects[@]}" \
    -lcublas -o "$qualification_build/structured-qualification"
  set +x
  "$qualification_cache" --show-stats > "$qualification_build/cache-after.txt"
  ldd "$qualification_build/structured-qualification" > "$qualification_build/ldd.txt"
  printf '%s\n' "$qualification_head" > "$qualification_build/source-head.txt"
  git -C "$qualification_root" status --porcelain --untracked-files=no > "$qualification_build/source-status.txt"
  git -C "$qualification_root" ls-files --stage > "$qualification_build/source-blobs-modes.txt"
  sha256sum "$qualification_build/structured-qualification" "$qualification_build/"*.o \
    "$qualification_build/generated_"* > "$qualification_build/artifact-sha256.txt"
fi
if [[ $qualification_mode = build-only ]]; then exit 0; fi
[[ $(cat "$qualification_build/source-head.txt") = "$qualification_head" ]]
[[ ! -s $qualification_build/source-status.txt ]]
printf 'Slurm job: %s\nCUDA_VISIBLE_DEVICES: %s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" \
  > "$qualification_build/slurm.txt"
date -u +%FT%TZ > "$qualification_build/run-start.txt"
nvidia-smi -q > "$qualification_build/nvidia-smi.txt"
"$qualification_build/structured-qualification" | tee "$qualification_build/receipt.json"
date -u +%FT%TZ > "$qualification_build/run-end.txt"

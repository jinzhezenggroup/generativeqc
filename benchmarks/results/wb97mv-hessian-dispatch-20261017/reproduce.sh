#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a finite single-GPU srun allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve Slurm-assigned device visibility}"
: "${SOURCE_CHECKOUT:?set a checkout containing the measured base}"
: "${NATIVE_ROOT:?set the measured linked native artifact directory}"
: "${REFERENCE_PYTHON:?set the recorded reference interpreter}"
bundle=$(cd -- "$(dirname -- "$0")" && pwd)
native=$(cd -- "$NATIVE_ROOT" && pwd)
output=${OUTPUT:-.artifacts/benchmarks/wb97mv-hessian-dispatch-$SLURM_JOB_ID.json}
# Preserve successful and failed receipts outside the disposable source overlay.
[[ $output = /* ]] || output=$PWD/$output
printf '%s  %s\n' \
  aa8b7480532767d2aa6de9a36a0fc70b3ff95b3606540e28388295fdd141d805 "$native/generativeqc" \
  4d06fe6a2182810670f49db5de3c0328415a722b109b210bdb152e4ff55ba0b4 "$native/libgenerativeqc.so.0.1.0" | sha256sum --check
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
git -C "$SOURCE_CHECKOUT" archive 4dae97170434d4fb697f721c2f1009666786ff47 | tar -xf - -C "$work"
xz -dc "$bundle/candidate-source.patch.xz" | git -C "$work" apply -
export PYTHONPATH="$work/python:$work/tests/python:$work"
export GENERATIVEQC_LIBRARY="$native/libgenerativeqc.so.0.1.0"
export LD_LIBRARY_PATH="$native:${REFERENCE_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DISABLE_PTX_JIT=1
cd "$work"
"$REFERENCE_PYTHON" benchmarks/compare_wb97mv_hessian_gpu4pyscf.py \
  --cases water-dimer formaldehyde water h2 --repeats 3 --timeout-seconds 180 \
  --gpu-memory-limit-mib 24576 --cli "$native/generativeqc" \
  --reference-python "$REFERENCE_PYTHON" --source-commit 4dae97170434d4fb697f721c2f1009666786ff47+working-tree \
  --output "$output"

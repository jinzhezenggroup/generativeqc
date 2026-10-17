#!/usr/bin/env bash
set -euo pipefail
: "${GENERATIVEQC_REFERENCE_PYTHON:?Set the pinned GPU4PySCF/PySCF reference interpreter}"
: "${GENERATIVEQC_CLI:?Set the source-built AOT native CLI path}"
: "${GENERATIVEQC_COMPILER_CACHE:?Set a verified sccache or ccache launcher}"
"$GENERATIVEQC_COMPILER_CACHE" --version
printf 'Native measured revision: 618f7e3ddf616f7960e9e3fc9df6aa31fda26107\n'
printf 'Retain all stock def2-SVP comparator failures; no speed qualification is expected there.\n'
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --time=00:12:00 "$GENERATIVEQC_REFERENCE_PYTHON" \
  benchmarks/compare_wb97mv_hessian_gpu4pyscf.py --cases water-tetramer \
  --basis sto-3g --engines generativeqc gpu4pyscf --repeats 1 \
  --timeout-seconds 180 --gpu-memory-limit-mib 24576 \
  --cli "$GENERATIVEQC_CLI" --reference-python "$GENERATIVEQC_REFERENCE_PYTHON" \
  --source-commit "$(git rev-parse HEAD)" --output .artifacts/capacity-tetramer.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --time=00:12:00 "$GENERATIVEQC_REFERENCE_PYTHON" \
  benchmarks/compare_wb97mv_hessian_gpu4pyscf.py --cases water-dimer \
  --basis def2-svp --engines generativeqc gpu4pyscf --repeats 1 \
  --timeout-seconds 180 --gpu-memory-limit-mib 24576 \
  --cli "$GENERATIVEQC_CLI" --reference-python "$GENERATIVEQC_REFERENCE_PYTHON" \
  --source-commit "$(git rev-parse HEAD)" --output .artifacts/capacity-def2svp.json

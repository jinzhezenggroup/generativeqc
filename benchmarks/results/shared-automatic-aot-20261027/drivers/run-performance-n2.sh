#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/tmp/qc-1123-shared-aot-20261027
export PATH="/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/pro6000-cuda/libgenerativeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR="$PWD/evidence/forces"
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 --time=02:00:00 python -u bench-suite.py > "evidence/performance-${PACKAGE_SCOPE:-narrow}.log" 2>&1

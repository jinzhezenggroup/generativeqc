#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/tmp/qc-1118-native-force-20261027
export PATH="/home/jzzeng/tmp/qc-1118-cuda-force-v9-20261011/.venv/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/native-cuda/libgenerativeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR="$PWD/evidence/forces"
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 bash -c 'set -e; ./build/native-cuda/generativeqc_dft_cuda_tests; ./build/native-cuda/generativeqc_ks_cuda_tests; GENERATIVEQC_AUTOMATIC_CUDA_KS_TEST=1 python -m pytest -x -q tests/python/test_libxc_automatic_cuda_ks.py; python -m pytest -x -q tests/python/test_libxc_native_cuda_forces.py -k "minimum_and_below_budget or ragged_failure or nonconvergence"' > evidence/final-regression.log 2>&1

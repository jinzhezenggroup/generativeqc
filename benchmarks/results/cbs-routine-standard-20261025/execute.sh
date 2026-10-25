#!/usr/bin/env bash
set -euo pipefail
task_root=/data/jzzeng/cbs-standard-20261025
export PATH=/home/jzzeng/.local/bin:/home/jzzeng/miniconda3/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
variant=${CBS_VARIANT:-candidate2}
export GENERATIVEQC_LIBRARY="${GENERATIVEQC_LIBRARY:-$task_root/$variant-library/libgenerativeqc.so}"
export GENERATIVEQC_PROFILE=off
export PYSCF_MAX_MEMORY=12000 PYTHONUNBUFFERED=1
export PYTHONPATH="$task_root/source/python:$task_root/source"
export GENERATIVEQC_FULL_DF_CUDA_TEST=1 GENERATIVEQC_CBS_CUDA_TEST=1
cd "$task_root/source"
exec python "$@"

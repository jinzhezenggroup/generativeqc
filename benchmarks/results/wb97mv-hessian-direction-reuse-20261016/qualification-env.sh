#!/usr/bin/env bash
set -euo pipefail
cd "$1"
shift
export CUDA_HOME=/group/software/cuda-12.9.1 CUDA_PATH=/group/software/cuda-12.9.1
export PATH="/data/jzzeng/qc-vv10-df-validation-20261015/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD:$CUDA_HOME/targets/x86_64-linux/lib:/home/jzzeng/codes/wb97m-20261002/gpu-env/lib/python3.13/site-packages/cutensor/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DISABLE_PTX_JIT=1
export GENERATIVEQC_LIBRARY="$PWD/libgenerativeqc.so.0.1.0" PYTHONPATH=python:tests/python:.
export GENERATIVEQC_CUDA_WB97MV_NUCLEAR_JK_PROBE="$PWD/generativeqc_cuda_wb97mv_nuclear_jk_probe"
export GENERATIVEQC_KS_WB97MV_HESSIAN_PROBE="$PWD/generativeqc_ks_wb97mv_hessian_probe"
export GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/generativeqc_ks_wb97mv_grid_probe"
export GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST=1 GENERATIVEQC_TEST_NATIVE_HESSIAN_FRONTEND=1
export GENERATIVEQC_WB97MV_CLI="$PWD/generativeqc"
exec "$@"

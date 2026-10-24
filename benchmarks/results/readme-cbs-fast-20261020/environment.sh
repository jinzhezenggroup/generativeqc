#!/usr/bin/env bash
set -euo pipefail
export CBS_FAST_ROOT=/data/jzzeng/qc-cbs-fast-mixed-20261020
export PATH=/home/jzzeng/.local/bin:/usr/local/bin:/home/jzzeng/codes/wb97m-20261002/ccache/usr/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH="$CBS_FAST_ROOT/runtime:/home/jzzeng/codes/wb97m-20261002/ccache/usr/lib/x86_64-linux-gnu:/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64"
export GENERATIVEQC_LIBRARY="$CBS_FAST_ROOT/runtime/libgenerativeqc.so"
export PYTHONPATH="$CBS_FAST_ROOT/source:$CBS_FAST_ROOT/source/python:/data/jzzeng/qc-j-opt-reviewed-final-20261020/deps"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export GENERATIVEQC_CBS_CUDA_TEST=1 GENERATIVEQC_FULL_DF_CUDA_TEST=1
export CBS_FAST_PYTHON=/data/jzzeng/qc-r2scan-force-branches-20261015/python.sh
cd "$CBS_FAST_ROOT/source"

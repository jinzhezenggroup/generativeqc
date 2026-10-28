set -euo pipefail
cd /home/jzzeng/tmp/qc-pr-unblock-2835-20261028
export PATH=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin
export LD_LIBRARY_PATH=$PWD/build/integration:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=python:tests/python:.
export GENERATIVEQC_LIBRARY=$PWD/build/integration/libgenerativeqc.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
python .artifacts/force-integration.py

#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?finite allocation required}"
: "${CUDA_VISIBLE_DEVICES:?preserve visibility}"
root=/data/jzzeng/gqc-direct-md-20261016
cd "$root/candidate-current/source"
export SOURCE="$PWD"
export PATH=/data/jzzeng/module-fast-fock-20261015/tools:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export GENERATIVEQC_LIBRARY="$root/candidate-current/runtime/libgenerativeqc.so"
export LD_LIBRARY_PATH="$root/candidate-current/runtime:/data/jzzeng/module-fast-fock-20261015/tools:/group/software/cuda-12.9.1/lib64"
export PYTHONPATH=.:python:tests/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
scontrol show job "$SLURM_JOB_ID" -o
sha256sum "$root/selected-j-oracle.py" "$GENERATIVEQC_LIBRARY"
timeout --kill-after=10 840 /data/jzzeng/qc-r2scan-force-branches-20261015/python.sh "$root/selected-j-oracle.py"

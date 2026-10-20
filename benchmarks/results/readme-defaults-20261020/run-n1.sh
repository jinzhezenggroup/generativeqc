#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/readme-defaults-20261020
export PATH=/home/jzzeng/.local/bin:/data/jzzeng/qc-provider-1886-tools/bin:/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:/home/jzzeng/miniconda3/lib"
export CCACHE=/home/jzzeng/.local/bin/sccache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/source"
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
"$CCACHE" --version
python -m benchmarks.readme_defaults "$1" --method "$2" --mode "$3" --atoms "$4" \
 --inputs "$root/inputs" --warm-repeats 3 \
 --source-revision e5fa4d2f0cd13775327261a0b6ced1eab6006c62 \
 --output "$5"

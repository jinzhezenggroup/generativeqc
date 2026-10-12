#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ragged-default-20261012
export PATH="/data/jzzeng/qc-ace-master-20261012/venv311/bin:/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="$root/source/build-cuda:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export GENERATIVEQC_LIBRARY="$root/source/build-cuda/libgenerativeqc.so"
export GENERATIVEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc
export GENERATIVEQC_STATIONARY_CACHE="$root/source/build-cuda"
export PYTHONPATH="$root/source/python:$root/source:$root/source/tests/python"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
printf 'Slurm job=%s visible=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
sccache --version
cd "$root/source"
python -u -m benchmarks.pbe0_ragged --atoms 96 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
  --reference benchmarks/results/pbe0-xc-tiles-20261006/reference-96.json.gz \
  --output "$root/validation/ragged-default-96.json" --repeats 5
python -u -m benchmarks.pbe0_ragged --atoms 48 \
  --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
  --reference benchmarks/results/pbe0-xc-tiles-20261006/reference-48.json.gz \
  --output "$root/validation/ragged-default-48.json" --repeats 5

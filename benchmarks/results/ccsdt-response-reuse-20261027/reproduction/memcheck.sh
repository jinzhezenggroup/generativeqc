#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/issue2516-20261027
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$root/evidence/qualified:$deps/lib:/group/software/cuda-12.9.1/lib64"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/source"
export PYTHONPATH="$PWD/python:$PWD:$deps" CCACHE_BASEDIR="$PWD"
export GENERATIVEQC_LIBRARY="$root/evidence/qualified/libgenerativeqc.so"
export GENERATIVEQC_DF_TRIPLES_CUDA_TEST=1
compute-sanitizer --tool memcheck --error-exitcode=99 python -m pytest -q tests/python/test_df_triples_scalar_fusion.py::test_fused_native_energy_tree_spans_multiple_ctas > "$root/evidence/memcheck-$SLURM_JOB_ID.log" 2>&1

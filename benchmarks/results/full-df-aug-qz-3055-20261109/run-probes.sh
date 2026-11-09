#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/qc-3055-full-df-final
srun --immediate=60 --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=20G --time=00:30:00 bash -lc '
set -euo pipefail
cd /home/jzzeng/codes/qc-3055-full-df-final
export PATH=$PWD/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH=python:.
export LD_LIBRARY_PATH=$PWD/native/blas:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export GENERATIVEQC_LIBRARY=$PWD/native/libgenerativeqc.so.0.1.0
export GENERATIVEQC_DF_G_CUDA_TEST=1
export GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export NVCC_PREPEND_FLAGS=-ccbin=/usr/bin/g++-11
printf "HOST=%s JOB=%s VISIBLE=%s\n" "$(hostname)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
sha256sum "$GENERATIVEQC_LIBRARY"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv
sccache --version
sccache --show-stats >cache-before.txt
/home/jzzeng/miniconda3/bin/python -m pytest -q -s tests/python/test_df_auxiliary_g.py tests/python/test_df_auxiliary_g_derivatives.py -k "cuda and (all_added or orbital_g_complete or diffuse_tight)"
sccache --show-stats >cache-after.txt
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 /home/jzzeng/miniconda3/bin/python -m pytest -q -s tests/python/test_one_electron_derivatives_cuda.py -k arbitrary_nonsymmetric
'

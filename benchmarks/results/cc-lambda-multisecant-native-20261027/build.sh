#!/usr/bin/env bash
set -euo pipefail
task_root=/data/jzzeng/cc-multisecant-20261027
export PATH=/home/jzzeng/.local/bin:/home/jzzeng/miniconda3/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task_root/source"
scontrol -d show job "$SLURM_JOB_ID" > "$task_root/build-allocation.txt"
sccache --version
sccache --show-stats > "$task_root/cache-before.txt"
cmake -S . -B build/cuda-release-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/jzzeng/miniconda3 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/home/jzzeng/.local/bin/sccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=/home/jzzeng/.local/bin/sccache \
  -DPython3_EXECUTABLE=/home/jzzeng/miniconda3/bin/python3.13 \
  -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=4 -DGENERATIVEQC_AOT_COMPILE_JOBS=4 \
  -DOpenBLAS_DIR=/data/jzzeng/direct-dft96-cold-20261023/stage22-blas-deps/scipy_openblas32/lib/cmake/openblas \
  -DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/cuda-release-sm120 --parallel 8 --target generativeqc
sccache --show-stats > "$task_root/cache-after.txt"
sha256sum build/cuda-release-sm120/libgenerativeqc.so > "$task_root/library.sha256"

#!/usr/bin/env bash
set -euo pipefail
task_root=/data/jzzeng/cbs-standard-20261025
variant=${1:-candidate}
export PATH=/home/jzzeng/.local/bin:/home/jzzeng/miniconda3/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task_root/source"
scontrol -d show job "$SLURM_JOB_ID" > "$task_root/${variant}-build-allocation.txt"
sccache --version
sccache --show-stats > "$task_root/cache-before-${variant}.txt"
cmake --build build/cuda-release-sm120 --parallel 8 --target generativeqc generativeqc_mean_field_options_tests
sccache --show-stats > "$task_root/cache-after-${variant}.txt"
mkdir -p "$task_root/${variant}-library"
cp -a build/cuda-release-sm120/libgenerativeqc.so* "$task_root/${variant}-library/"
sha256sum "$task_root/${variant}-library/"* > "$task_root/${variant}-library.sha256"
build/cuda-release-sm120/generativeqc_mean_field_options_tests
python - <<'PY'
import ctypes
from pathlib import Path
library=ctypes.CDLL(str(Path('build/cuda-release-sm120/libgenerativeqc.so').resolve()))
query=library.generativeqc_correlation_execution_version_v1
query.argtypes=[]
query.restype=ctypes.c_uint32
assert query() == 1
print('Independent execution capability: 1', flush=True)
PY

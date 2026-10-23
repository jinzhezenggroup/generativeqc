#!/usr/bin/env bash
# One bounded network repair after the retained PyPI bootstrap attempt.
set -euo pipefail
root=/data/srxu/projects/gqc-runtime-cache-state-20261023
cd "$root"
: "${SLURM_JOB_ID:?requires real CPU Slurm allocation}"
test "$(id -un)" = srxu
test ! -e receipts/resume-started
test -f receipts/source-verified.json
test -x runtime-env/bin/python
printf '%s\n' "$SLURM_JOB_ID" > receipts/resume-started
trap 'rc=$?; printf "%s\n" "$rc" > receipts/resume-exit-code' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
/usr/bin/scontrol show job "$SLURM_JOB_ID" > receipts/resume-allocation.txt
# Source and the private partially bootstrapped environment stay in place.
# This is the documented USTC public PyPI mirror; no global configuration,
# package cache, source identity or production option is changed.
/home/srxu/miniconda3/bin/python -m pip --isolated --python "$root/runtime-env/bin/python" install \
  --index-url https://mirrors.ustc.edu.cn/pypi/simple pip \
  'numpy==2.4.6' 'scipy>=1.10' 'pyscf==2.14.0' 'cupy-cuda12x==13.6.0' \
  'gpu4pyscf-cuda12x==1.8.1' 'sccache==0.16.0' 'cmake>=3.28,<5' ninja \
  'array-api-compat>=1.15.0' typing-extensions 'jinja2>=3' sympy mpmath \
  'nvidia-cublas-cu12>=12.9.1.4' 'nvidia-cusolver-cu12>=11.7.5.82' \
  'nvidia-cusparse-cu12>=12.5.10.65' 'nvidia-cuda-runtime-cu12>=12.9.79' \
  'nvidia-nvjitlink-cu12>=12.9.86'
export PATH="$root/runtime-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export PYTHONPATH="$root/source/python:$root/source"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SCCACHE_DIR="$root/compiler-cache"
export SCCACHE_CONF="$root/compiler-cache.toml"
export SCCACHE_CACHED_CONF="$root/compiler-cache-cached.toml"
export SCCACHE_SERVER_UDS="$root/compiler-cache.sock"
export CCACHE_BASEDIR="$root/source"
printf '' > "$SCCACHE_CONF"
sccache --version > receipts/compiler-cache-version.txt
python -c 'import re; from pathlib import Path; v=Path("receipts/compiler-cache-version.txt").read_text(); m=re.search(r"sccache (\d+)\.(\d+)\.(\d+)",v); assert m and tuple(map(int,m.groups())) >= (0,16,0)'
sccache --show-stats > receipts/compiler-cache-before.txt
cmake -S source -B source/build/cuda-release-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/sdk" \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$root/runtime-env/bin/sccache" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$root/runtime-env/bin/sccache" \
  -DPython3_EXECUTABLE="$root/runtime-env/bin/python" \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_AOT_PROFILE=sm_120 \
  -DGENERATIVEQC_CUDA_ARCHITECTURES=120 -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
  -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=ON \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=1 -DGENERATIVEQC_CUDA_COMPILE_JOBS=2
ninja -C source/build/cuda-release-sm120 -t commands > receipts/compiler-commands.txt
cmake --build source/build/cuda-release-sm120 --parallel 4
sccache --show-stats > receipts/compiler-cache-after.txt
cmake --install source/build/cuda-release-sm120
runtime-env/bin/python -m pip freeze > receipts/packages.txt
printf 'built; device qualification NOT_RUN\n' > receipts/build-complete

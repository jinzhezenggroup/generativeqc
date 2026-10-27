#!/usr/bin/env bash
# Runtime environment for GenerativeQC on USTC SCC 瀚海22 (A100) compute nodes.
# Source this file from an sbatch script; do not execute it directly.
#
# Requires:
#   - CUDA 12.8 + gcc 13.1.0 + cmake 3.30.5 + python 3.12.2 modules
#   - a venv OUTSIDE $HOME (home quota is small; the Python stack is several GB)
#   - GENERATIVEQC_LIBRARY pointing at an sm_80 build, never the sm_120 default
#
# The source root is derived from this file's location, so the checkout can live
# anywhere. Override GQC_VENV / GQC_BUILD to relocate outputs.

# This file is sourced from `set -euo pipefail` sbatch scripts, so every
# expansion below must stay safe under `nounset`.
_gqc_scc_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export GQC_SRC="${GQC_SRC:-$(cd -- "${_gqc_scc_dir}/../.." && pwd)}"
export GQC_VENV="${GQC_VENV:-${GQC_SRC}/.venv-scc}"
export GQC_BUILD="${GQC_BUILD:-${GQC_SRC}/build/cuda-release-sm80}"

module purge
module load cuda/12.8
module load gcc/13.1.0
module load cmake/3.30.5
module load python/3.12.2

# One process per benchmark unit: never let BLAS/OMP compete for the cores we
# were allocated, since the timings are single-device endpoint measurements.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Derive CUDA_PATH from nvcc rather than hardcoding a site-specific prefix.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found after module load" >&2
  return 1
fi
export CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDACXX="$CUDA_PATH/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

export GENERATIVEQC_LIBRARY="$GQC_BUILD/libgenerativeqc.so"

# Both entries are required:
#   $GQC_SRC        -> the `benchmarks` package (readme_defaults.py imports
#                      `benchmarks._retention` at module scope)
#   $GQC_SRC/python -> the generativeqc Python bindings
# Providing only python/ raises ModuleNotFoundError for `benchmarks`, because a
# script run as __main__ puts its own directory (benchmarks/) on sys.path[0],
# not the source root.
export PYTHONPATH="$GQC_SRC:$GQC_SRC/python${PYTHONPATH:+:$PYTHONPATH}"

export PATH="$GQC_VENV/bin:$CUDA_PATH/bin:$PATH"

# Do not inherit a debug override from an interactive shell: a stale response
# budget changes DF memory planning and would silently alter the measurement.
unset GENERATIVEQC_DF_RESPONSE_BUDGET_BYTES 2>/dev/null || true

echo "[env] node   = $(hostname)"
echo "[env] src    = $GQC_SRC"
echo "[env] nvcc   = $(command -v nvcc) ($(nvcc --version | tail -2 | head -1 | tr -s ' '))"
echo "[env] cuda   = $CUDA_PATH"
echo "[env] gcc    = $(gcc --version | head -1)"
echo "[env] cmake  = $(cmake --version | head -1)"
echo "[env] python = $(command -v python3)"
echo "[env] venv   = $GQC_VENV"
echo "[env] lib    = $GENERATIVEQC_LIBRARY"

unset _gqc_scc_dir

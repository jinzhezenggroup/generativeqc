#!/usr/bin/env bash
# Execute this entry point under srun in the qz CPU HPC allocation.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cpu_python=${GQC_REFERENCE_PYTHON:?absolute path to the prepared PySCF interpreter}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
exec "$cpu_python" -u "$root/run_reference.py" campaign "$@"

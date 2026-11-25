#!/usr/bin/env bash
# Execute under srun in the CPU HPC allocation.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=${1:?new output directory}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
exec "$root/cpu-env/bin/python" -u "$root/run_reference.py" campaign "$output"

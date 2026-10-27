#!/bin/bash
# Run the water-96 standard-arm benchmark cases on a USTC SCC GPU node.
#
#   sbatch tools/scc/run.sbatch
#   OUT=/scratch/<user>/gqc-a100 sbatch tools/scc/run.sbatch
#
# V100 (瀚海20, sm_70) — the library must already be built for sm_70:
#   sbatch --partition=<v100-partition> --gres=gpu:v100:1 \
#          --export=ALL,GQC_ARCH=70 tools/scc/run.sbatch
#
# The account is allocation-specific, so it is not hardcoded here. Either add
#   #SBATCH -A <ACCOUNT>
# or pass --account=<ACCOUNT> on the command line.
#
#SBATCH -J gqc-run
#SBATCH -o gqc-run-%j.out
#SBATCH -e gqc-run-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Why one GPU and 8 CPUs: four serial water-96 cases, one process each, with
# single-threaded BLAS/OMP so the timings are single-device endpoint numbers.
# On an RTX 5090 the four cases total roughly 2 minutes; 1 hour is ample even
# if A100 is slower.
#
# Why 64 GB: the DF cases allocate a metric plus a contraction tile, and too
# little memory fails with
#   "DF memory budget cannot hold the metric and one contraction tile"
# The node has 1 TB, so this stays far below the node limit and does not
# affect queueing.
#
# Do not run the benchmark on a login node: benchmarks/readme_defaults.py
# refuses real GPU work outside a finite Slurm GPU allocation.

set -euo pipefail
echo "job=${SLURM_JOB_ID:-local} node=$(hostname) $(date)"
echo "gres=${SLURM_GRES:-none} cpus=${SLURM_CPUS_PER_TASK:-unset} arch=${GQC_ARCH:-80}"
echo "src=${SLURM_SUBMIT_DIR:-$(pwd)}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
echo

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "${ROOT}/tools/scc/run.sh" ]; then
    echo "ERROR: ${ROOT}/tools/scc/run.sh not found." >&2
    echo "       Submit this script from the repository root (sbatch tools/scc/run.sbatch)." >&2
    exit 1
fi

# Default output stays inside the checkout (git-ignored). Results are small, but
# redirect OUT when the AOT build has already consumed the home quota.
OUT="${OUT:-${ROOT}/.artifacts/scc-a100-sm${GQC_ARCH:-80}}"
echo "results=${OUT}"
echo

bash "${ROOT}/tools/scc/run.sh" "${OUT}"

echo
echo "run 完成 $(date)"

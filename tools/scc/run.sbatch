#!/bin/bash
# Run the water-96 benchmark cases for ONE arm on a USTC SCC GPU node.
#
#   # GenerativeQC Standard arm (the implementation under test)
#   sbatch tools/scc/run.sbatch
#
#   # GPU4PySCF reference arm, same methods/modes/gates
#   sbatch --export=ALL,GQC_ARM=gpu4pyscf tools/scc/run.sbatch
#
#   # group 1: the cheap calibration set (hf + pbe, direct + df)
#   sbatch tools/scc/run.sbatch
#
#   # group 2: the paths the reference RTX 5090 cohort showed to be slow/broken
#   sbatch --export=ALL,GQC_GROUP=defects tools/scc/run.sbatch
#
#   OUT=/scratch/<user>/gqc-a100 sbatch tools/scc/run.sbatch
#
# A standard/GPU4PySCF pair for the same group is two submissions with the same
# GQC_GROUP and different GQC_ARM; results land in separate directories and
# tools/scc/compare_scc.py joins them.
#
# V100 (瀚海20, sm_70) — the library must already be built for sm_70:
#   sbatch --partition=<v100-partition> --gres=gpu:v100:1 \
#          --export=ALL,GQC_ARCH=70 tools/scc/run.sbatch
#
# The account is allocation-specific, so it is not hardcoded here. Either add
#   #SBATCH -A <ACCOUNT>
# or pass --account=<ACCOUNT> on the command line.
#
# --qos is not optional on this site. Each partition admits exactly one QoS
# (GPU-8A100 admits gpu_8a100), and the account has no DefaultQOS, so omitting
# it fails with "Invalid qos specification" even when -p and -A are correct.
# When you override -p you must override --qos too.
#
#SBATCH -J gqc-run
#SBATCH -o gqc-run-%j.out
#SBATCH -e gqc-run-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Why -n 1 -c 8 rather than -n 8: the cases run serially, one process each, so a
# single task holds the cores. Asking for -n 8 with --cpus-per-task=8 requests
# 64 cores, the whole node, which is both unnecessary and bad for queueing.
# BLAS/OMP stay single-threaded so the timings are single-device endpoints.
#
# Why 4 hours: the baseline group is minutes, but the defects group contains
# wb97m-v, which on an RTX 5090 took 1130 s for a single cold evaluation with 39
# SCF iterations. Multiplied by cold + 3 warm repeats, and allowing for the A100
# being slower per iteration, a 1-hour limit would be cut off partway through
# the most interesting case. run.sh additionally caps each individual case via
# GQC_TIMEOUT so one pathological method cannot eat the whole slot.
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
echo "arm=${GQC_ARM:-standard} group=${GQC_GROUP:-baseline}"
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
# Arm and group are both in the path so the two arms never overwrite each
# other's pbe:df reference point, and the two groups never do either.
OUT="${OUT:-${ROOT}/.artifacts/scc-a100-sm${GQC_ARCH:-80}/${GQC_ARM:-standard}/${GQC_GROUP:-baseline}}"
echo "results=${OUT}"
echo

bash "${ROOT}/tools/scc/run.sh" "${OUT}"

echo
echo "run 完成 $(date)"

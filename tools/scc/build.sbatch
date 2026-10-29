#!/bin/bash
# Build GenerativeQC for a USTC SCC GPU architecture on a compute node.
#
#   sbatch tools/scc/build.sbatch
#
# V100 (瀚海20, sm_70) — override both the partition/GRES and the arch:
#   sbatch --partition=<v100-partition> --gres=gpu:v100:1 \
#          --export=ALL,GQC_ARCH=70 tools/scc/build.sbatch
#
# The account is allocation-specific, so it is not hardcoded here. Either add
#   #SBATCH -A <ACCOUNT>
# or pass --account=<ACCOUNT> on the command line. Removing -p makes Slurm use
# the account's default partition.
#
# --qos is not optional on this site. Each partition admits exactly one QoS
# (GPU-8A100 admits gpu_8a100, CPU-64C256GB admits qos_cpu_64c256gb), and the
# account has no DefaultQOS, so omitting --qos fails with
#   "Invalid qos specification"
# even when -p and -A are correct. The default here matches -p GPU-8A100; when
# you override -p you must override --qos too.
#
#SBATCH -J gqc-build
#SBATCH -o gqc-build-%j.out
#SBATCH -e gqc-build-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Why one task with 16 cores: the build is one ninja job tree, so -n 1 keeps the
# request honest while -c 16 gives that task its threads. Requesting -n 16 with
# --cpus-per-task=16 asks for 384 cores on a 64-core node and Slurm rejects it
# with "Requested node configuration is not available".
#
# Why one GPU: CMake's CUDA configuration probes the visible device for its SM
# count, so a GPU-less allocation fails at configure time. It is not used for
# compute; the compile targets come from GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES.
#
# Why 16 CPUs / 64 GB: the AOT stable-shards step builds several shards in
# parallel, and a cold build (venv + full compile) is the memory-heaviest phase.
# The node has 1 TB, so this is generous without affecting queueing.

set -euo pipefail
echo "job=${SLURM_JOB_ID:-local} node=$(hostname) $(date)"
echo "gres=${SLURM_GRES:-none} cpus=${SLURM_CPUS_PER_TASK:-unset} arch=${GQC_ARCH:-80}"
echo "src=${SLURM_SUBMIT_DIR:-$(pwd)}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
echo

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "${ROOT}/tools/scc/build.sh" ]; then
    echo "ERROR: ${ROOT}/tools/scc/build.sh not found." >&2
    echo "       Submit this script from the repository root (sbatch tools/scc/build.sbatch)." >&2
    exit 1
fi

# Read-only on the source tree: everything lands in build/ and .venv-scc/,
# both of which are git-ignored, so the checkout stays clean for commits.
JOBS="${JOBS:-16}" bash "${ROOT}/tools/scc/build.sh"

echo
echo "build 完成 $(date)"

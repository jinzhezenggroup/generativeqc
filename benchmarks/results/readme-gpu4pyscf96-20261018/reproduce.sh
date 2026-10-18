#!/usr/bin/env bash
# Run only the two missing independent 96-atom populations, without a process deadline.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun with an explicit finite allocation}"
: "${CUDA_VISIBLE_DEVICES:?Preserve Slurm-assigned device visibility}"
root="${README_GPU4PYSCF96_ROOT:?Select the retained campaign directory}"
mode="${1:?Select none (Direct) or cuda (DF)}"
case "$mode" in none|cuda) ;; *) exit 2 ;; esac
cd "$root/source"
export PATH=/home/jzzeng/codes/wb97m-20261002/gpu-env/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export LD_PRELOAD="$CUDA_PATH/lib64/libcudart.so.12:$CUDA_PATH/lib64/libcublas.so.12:$CUDA_PATH/lib64/libcusolver.so.11"
export PYTHONPATH=.:python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
directory="$root/measurements/$mode/wb97m-v/96"
mkdir -p "$directory" "$root/receipts"
test ! -e "$directory/reference.json"
date --iso-8601=ns
scontrol show job "$SLURM_JOB_ID" -o | tee "$root/receipts/$SLURM_JOB_ID.job.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name --format=csv
extra=()
if [[ "$mode" == cuda ]]; then
    extra+=(--auxiliary-basis-file benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json)
fi
result=0
python -m benchmarks.readme_dft --method wb97m-v reference \
    --atoms 96 --basis-file "$root/inputs/def2-svp.json" \
    --density-fitting "$mode" --repeats 5 "${extra[@]}" \
    --output "$directory/reference.json" > "$directory/reference.log" 2>&1 || result=$?
printf '{"exit_code":%s,"time_limit_seconds":null}\n' "$result" > "$directory/reference.outcome"
date --iso-8601=ns
exit "$result"

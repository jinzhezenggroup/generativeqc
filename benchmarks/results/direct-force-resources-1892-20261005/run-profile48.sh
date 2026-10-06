#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?finite Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm-assigned visibility required}"
if [[ ${SLURM_STEP_ID:-batch} == batch ]]; then
  exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --time=00:28:00 bash "$0" "$@"
fi
case "$SLURM_STEP_ID" in extern) exit 2 ;; esac
variant=${1:?64 or 128 required}
case "$variant" in 64|128) ;; *) exit 2 ;; esac
root=/data/jzzeng/qc-1892-p0b-20261005
candidate=/data/jzzeng/qc-1892-p0b-cta-20261005
test "$(cat "$candidate/.artifacts/qualification/6091/job.exit")" = 0
test "$(cat "$candidate/.artifacts/triage/6095/job.exit")" = 0
assets=$root/endpoint-assets
cuda=/group/software/cuda-12.9.1
environment=/home/jzzeng/codes/wb97m-20261002
python=$environment/gpu-env/bin/python
nsys=$cuda/nsight-systems-2025.1.3/target-linux-x64/nsys
output=$candidate/.artifacts/profiles/$SLURM_JOB_ID
mkdir -p "$output"
trap 'printf "%s\n" "$?" > "$output/job.exit"' EXIT
export PATH="$assets/bin:$cuda/bin:$environment/ccache/usr/bin:$PATH"
export GENERATIVEQC_NVCC="$assets/bin/nvcc" CUDACXX="$assets/bin/nvcc"
export CXX="$assets/bin/c++" CUDA_PATH="$cuda"
export LD_LIBRARY_PATH="$environment/ccache/usr/lib/x86_64-linux-gnu:$environment/gpu-env/lib/python3.13/site-packages/cutensor/lib:$cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$cuda/lib64/libcudart.so.12:$cuda/lib64/libcublas.so.12:$cuda/lib64/libcublasLt.so.12:$cuda/lib64/libcusolver.so.11:$cuda/lib64/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
printf 'job=%s\nstep=%s\nvisible=%s\nvariant=%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$CUDA_VISIBLE_DEVICES" "$variant" > "$output/environment.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$output/job.txt"
scontrol show step "$SLURM_JOB_ID.$SLURM_STEP_ID" -o > "$output/step.txt"
cp "$0" "$assets/profile-single-warm.py" "$output/"
sha256sum "$output"/{run-profile48.sh,profile-single-warm.py} > "$output/drivers.sha256"
for arm in baseline default "$variant"; do
  checkout=$candidate
  threads=$arm
  receipt=.artifacts/build/binaries.sha256
  if [[ $arm == default ]]; then threads=0; fi
  if [[ $arm == baseline ]]; then checkout=$root/control; threads=0; receipt=.artifacts/control/binaries.sha256; fi
  cd "$checkout"
  sha256sum -c "$receipt" > "$output/$arm-binary-check.txt"
  export GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS=$threads
  export GENERATIVEQC_LIBRARY="$checkout/build/cuda-release-sm120/libgenerativeqc.so"
  export PYTHONPATH="$checkout/python:$checkout" CCACHE_BASEDIR="$checkout"
  printf '%s START %s\n' "$(date -Is)" "$arm" | tee -a "$output/progress.txt"
  timeout 450 "$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$output/$arm" "$python" "$output/profile-single-warm.py" \
    --method pbe0 --atoms 48 --reference "$candidate/.artifacts/triage/6095/reference-48.json" \
    --output="$output/$arm.json" > "$output/$arm.log" 2>&1
  "$nsys" export --type=sqlite --output="$output/$arm.sqlite" "$output/$arm.nsys-rep" > "$output/$arm-export.log" 2>&1
  printf '%s DONE %s\n' "$(date -Is)" "$arm" | tee -a "$output/progress.txt"
done
sha256sum "$output"/*.json "$output"/*.sqlite "$output"/*.nsys-rep > "$output/evidence.sha256"

#!/usr/bin/env bash
set -euo pipefail

# The batch allocation is only a reservation; all device work runs in its
# explicit finite srun step, preserving the scheduler's assigned visibility.
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm-assigned visibility required}"
if [[ ${SLURM_STEP_ID:-batch} == batch ]]; then
  exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --time=01:55:00 bash "$0" "$@"
fi

mode=${1:?smoke or endpoints required}
case "$mode" in smoke|endpoints) ;; *) exit 2 ;; esac
root=/data/jzzeng/qc-1892-p0b-20261005
control=$root/control
candidate=/data/jzzeng/qc-1892-p0b-cta-20261005
assets=$root/endpoint-assets
cuda=/group/software/cuda-12.9.1
environment=/home/jzzeng/codes/wb97m-20261002
python=$environment/gpu-env/bin/python
test "$(cat "$candidate/.artifacts/qualification/6091/job.exit")" = 0
test "$(cat "$candidate/.artifacts/triage/6095/job.exit")" = 0
output=$candidate/.artifacts/campaigns/${SLURM_JOB_ID}-$mode
mkdir -p "$output"
trap 'printf "%s\n" "$?" > "$output/job.exit"' EXIT

export PATH="$assets/bin:$cuda/bin:$environment/ccache/usr/bin:$PATH"
export GENERATIVEQC_NVCC="$assets/bin/nvcc" CUDACXX="$assets/bin/nvcc"
export CXX="$assets/bin/c++" CUDA_PATH="$cuda"
export LD_LIBRARY_PATH="$environment/ccache/usr/lib/x86_64-linux-gnu:$environment/gpu-env/lib/python3.13/site-packages/cutensor/lib:$cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$cuda/lib64/libcudart.so.12:$cuda/lib64/libcublas.so.12:$cuda/lib64/libcublasLt.so.12:$cuda/lib64/libcusolver.so.11:$cuda/lib64/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1

scontrol show job "$SLURM_JOB_ID" -o > "$output/job.txt"
scontrol show step "$SLURM_JOB_ID.$SLURM_STEP_ID" -o > "$output/step.txt"
printf 'node=%s\njob=%s\nstep=%s\nvisible=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$CUDA_VISIBLE_DEVICES" \
  > "$output/environment.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$output/devices.txt"
"$python" - > "$output/allocated-runtime-device.json" <<'PY'
import json
import cupy as cp
properties = cp.cuda.runtime.getDeviceProperties(0)
print(json.dumps({
    "process_local_device": 0,
    "pci_bus_id": cp.cuda.Device(0).pci_bus_id,
    "runtime_uuid": repr(properties.get("uuid")),
    "visible_device_count": cp.cuda.runtime.getDeviceCount(),
}, indent=2))
PY
"$environment/ccache/usr/bin/ccache" --version > "$output/ccache-version.txt"
"$environment/ccache/usr/bin/ccache" --show-stats > "$output/ccache-before.txt"
cp "$assets/run-endpoint-arm.py" "$0" "$output/"
sha256sum "$output"/{run-endpoint-arm.py,run-matched-matrix.sh} > "$output/drivers.sha256"

for checkout in "$control" "$candidate"; do
  cd "$checkout"
  sha256sum -c "$assets/compiler-assets.sha256" > "$output/$(basename "$checkout")-assets-check.txt"
done
cd "$control"
sha256sum -c .artifacts/control/binaries.sha256 > "$output/control-binary-check.txt"
cd "$candidate"
sha256sum -c .artifacts/build/binaries.sha256 > "$output/candidate-binary-check.txt"
sha256sum -c .artifacts/build/source-files.sha256 > "$output/candidate-source-check.txt"

run_arm() {
  local arm=$1 method=$2 atoms=$3 engine=$4 reference=${5:-}
  local checkout=$control
  local enabled=0
  local snapshot_arm=control
  if [[ $arm == candidate ]]; then checkout=$candidate; enabled=128; snapshot_arm=candidate; fi
  cd "$checkout"
  export PYTHONPATH="$PWD/python:$PWD"
  export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
  export GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS=$enabled
  export CCACHE_BASEDIR="$PWD"
  export P0B_SNAPSHOT_IDENTITY="$assets/$snapshot_arm-source-identity.json"
  if [[ $arm == candidate ]]; then
    export P0B_SNAPSHOT_IDENTITY="$candidate/.artifacts/build/source-identity.json"
  fi
  local basename=$method-$atoms-$arm
  local destination=$output/$basename.json
  local repeats=5
  if [[ $mode == smoke ]]; then repeats=1; fi
  local command=("$python" "$output/run-endpoint-arm.py" --method "$method" "$engine")
  if [[ $method == pbe0 ]]; then
    command+=(--atoms "$atoms" --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats "$repeats" --output "$destination")
  else
    command+=(--aos "$((atoms * 8))" --nested-water --route direct --repeats "$repeats" --output "$output/$basename")
  fi
  if [[ -n $reference ]]; then command+=(--reference "$reference"); fi
  printf '%s START %s\n' "$(date -Is)" "$basename" | tee -a "$output/progress.txt"
  nvidia-smi --query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader > "$output/$basename-device-before.txt"
  timeout 2700 "${command[@]}" > "$output/$basename.log" 2>&1
  printf '%s DONE %s\n' "$(date -Is)" "$basename" | tee -a "$output/progress.txt"
}

if [[ $mode == smoke ]]; then
  run_arm reference pbe0 3 reference
  oracle=$output/pbe0-3-reference.json
  run_arm control pbe0 3 native "$oracle"
  run_arm candidate pbe0 3 native "$oracle"
else
  for method in hf pbe0; do
    for atoms in 48 96; do
      run_arm reference "$method" "$atoms" reference
      oracle=$output/$method-$atoms-reference.json
      if [[ $method == hf ]]; then oracle=$output/$method-$atoms-reference/results.json; fi
      # Reverse the second size's process order without changing scientific work.
      if [[ $atoms == 48 ]]; then arms=(control candidate); else arms=(candidate control); fi
      for arm in "${arms[@]}"; do
        run_arm "$arm" "$method" "$atoms" native "$oracle"
      done
    done
  done
fi
"$environment/ccache/usr/bin/ccache" --show-stats > "$output/ccache-after.txt"

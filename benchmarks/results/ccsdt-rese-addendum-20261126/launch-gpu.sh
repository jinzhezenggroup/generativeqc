#!/usr/bin/env bash
set -euo pipefail
root=$PWD
case_id=$1
run="$root/gpu-$case_id"
mkdir "$run"
date -u +%FT%TZ > "$run/entry.txt"
trap 'printf "%s\n" "$?" > "$run/exit-code.txt"; date -u +%FT%TZ > "$run/exit.txt"' EXIT
base="$root/../issue-3166-20261125-master22f"
envroot="$root/../h100-portability/20261027-fd06/env"
deps="$root/../issue-3166-20261125/cli-deps"
cuda=/opt/generativeqc-reuse/cuda-12.9
export LD_LIBRARY_PATH="$base/c/build:$deps/usr/lib/x86_64-linux-gnu:$envroot/lib/python3.11/site-packages/scipy_openblas32/lib:$cuda/lib64:$cuda/targets/x86_64-linux/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:16:8
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > "$run/gpu.txt"
lscpu > "$run/cpu.txt"
"$envroot/bin/python" -u "$root/relink-trace.py" "$base" "$run/bin" > "$run/relink.log" 2>&1
sha256sum "$run/bin/generativeqc" "$base/c/build/libgenerativeqc.so" > "$run/binaries.sha256"
"$envroot/bin/python" -u "$base/payload-rescue/run.py" "$base/payload-rescue" "$run/bin/generativeqc" "$run/raw" \
  --molecule "$case_id" --df-auxiliary-basis "$base/payload-rescue/cc-pwcvtz-rifit.json" --target-timeout 28800

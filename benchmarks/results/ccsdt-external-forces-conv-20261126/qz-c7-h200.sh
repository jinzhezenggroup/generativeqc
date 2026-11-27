#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
label=$1
output=$2
mkdir -p "$output"
{
    nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv
    awk '/^CMAKE_CUDA_ARCHITECTURES:|^CMAKE_BUILD_TYPE:/ {print}' "$GQC_NATIVE_BUILD/CMakeCache.txt"
    if [[ -x "$GQC_CUDA_ROOT/bin/cuobjdump" ]]; then
        "$GQC_CUDA_ROOT/bin/cuobjdump" --list-elf "$GQC_NATIVE_BUILD/libgenerativeqc.so" \
            | awk '/sm_90/ {sn++} END {print "SM90 ELF images:", sn+0; exit !sn}'
    fi
} > "$output/$label-aot.txt" 2>&1

bash "$root/qz-gpu-samples.sh" "$output/$label.gpu.csv" \
    bash "$root/qz-run.sh" native-conv "$label" "$output/$label" \
    --expected-gpu H200 --threads 1 --correlation-budget-gib 155 --timeout 46800

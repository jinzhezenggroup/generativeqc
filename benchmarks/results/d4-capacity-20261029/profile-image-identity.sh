#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-d4-image-identity-3643d2665-20261029
base_root=/home/jzzeng/codes/qc-2227-d4-capacity-6dbc152db-20261029
library="$base_root/artifacts/libgenerativeqc.so.0"
out="$run_root/artifacts"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export LD_LIBRARY_PATH="$base_root/artifacts:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 D4_QUALIFY_GLOBAL_BUDGET=1
printf 'Slurm job: %s; node: %s; assigned devices: %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
cd "$run_root"
sha256sum -c artifacts/input-sha256.txt > artifacts/integrity-before.log
python "$out/qualification/qualify.py" --baseline "$base_root" --candidate "$run_root" --baseline-library "$library" --candidate-library "$library" --backend cuda --groups 2 --output "$out/image-identity-diagnostic.json.xz"
python -m cProfile -o "$out/candidate.pstats" "$out/qualification/endpoint_worker.py" --source "$run_root" --library "$library" --backend cuda --once > "$out/candidate-profile.json"
sha256sum -c artifacts/input-sha256.txt > artifacts/integrity-after.log

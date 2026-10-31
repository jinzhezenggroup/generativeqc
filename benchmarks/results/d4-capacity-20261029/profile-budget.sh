#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-d4-capacity-6dbc152db-20261029
base_root=/home/jzzeng/codes/qc-2227-d4-ledger-4fc8782cf-20261029
out=/home/jzzeng/codes/qc-2227-d4-budget-profile-20261029
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export LD_LIBRARY_PATH="$run_root/artifacts:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 D4_QUALIFY_GLOBAL_BUDGET=1
printf 'Slurm job: %s; node: %s; assigned devices: %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
for role in baseline candidate; do
  source_root="$run_root"
  library="$run_root/artifacts/libgenerativeqc.so.0"
  if [ "$role" = baseline ]; then
    source_root="$base_root"
    library="$base_root/artifacts/libgenerativeqc.so.0"
  fi
  python -m cProfile -o "$out/$role.pstats" "$run_root/artifacts/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$out/$role-python-profile.json"
  prefix="$out/$role-profile"
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=false --output="$prefix" python "$run_root/artifacts/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$prefix.log" 2>&1
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$prefix-stats" "$prefix.nsys-rep"
done
python "$run_root/artifacts/qualification/profile_summary.py" --baseline "$out/baseline-profile.sqlite" --candidate "$out/candidate-profile.sqlite" --output "$out/profile-summary.json"

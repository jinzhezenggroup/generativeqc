#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-named-guard-9ce189713-20261029
base_root=/home/jzzeng/codes/qc-2227-d4-ledger-4fc8782cf-20261029
reference_root=/home/jzzeng/codes/qc-2227-precision-policy-20261029
out="$run_root/artifacts"
cd "$run_root"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH="$run_root/python:$run_root:$reference_root/artifacts/reference-python-v2:$out/qualification"
export GENERATIVEQC_LIBRARY="$out/libgenerativeqc.so.0"
export GENERATIVEQC_RESOURCE_CUDA_TEST=1 GENERATIVEQC_DFT_CUDA_TEST=1
export LD_LIBRARY_PATH="$out:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_PATH=/group/software/cuda-12.9.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Slurm job: %s; node: %s; assigned CUDA devices: %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm-running.txt"
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-before.log"
sccache --version
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
"$out/generativeqc_ks_capacity_cuda_tests"
env -i PATH=/nonexistent CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SLURM_JOB_ID="$SLURM_JOB_ID" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$out/named-sdk"
env -i PATH=/nonexistent CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SLURM_JOB_ID="$SLURM_JOB_ID" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$out/d4-sdk" cuda
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 "$out/named-sdk" > "$out/memcheck.log" 2>&1
python -m pytest tests/python/test_d4_production.py tests/python/test_parameterized_d4_composition.py tests/python/test_d3_d4_composition_compatibility.py tests/python/test_d3_calculator_composition.py --junitxml="$out/gpu-tests.xml" -q
python "$out/qualification/qualify.py" --baseline "$base_root" --candidate "$run_root" --baseline-library "$base_root/artifacts/libgenerativeqc.so.0" --candidate-library "$out/libgenerativeqc.so.0" --backend cuda --groups 8 --output "$out/cuda-samples.json.xz"
for role in baseline candidate; do
  source_root="$run_root"
  library="$out/libgenerativeqc.so.0"
  if [ "$role" = baseline ]; then
    source_root="$base_root"
    library="$base_root/artifacts/libgenerativeqc.so.0"
  fi
  prefix="$out/$role-profile"
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=false --output="$prefix" python "$out/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$prefix.log" 2>&1
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$prefix-stats" "$prefix.nsys-rep"
done
python "$out/qualification/profile_summary.py" --baseline "$out/baseline-profile.sqlite" --candidate "$out/candidate-profile.sqlite" --output "$out/profile-summary.json"
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-after.log"

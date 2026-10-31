#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-d4-capacity-6dbc152db-20261029
base_root=/home/jzzeng/codes/qc-2227-d4-ledger-4fc8782cf-20261029
reference_root=/home/jzzeng/codes/qc-2227-precision-policy-20261029
out="$run_root/artifacts"
cd "$run_root"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH="$run_root/python:$run_root:$reference_root/artifacts/reference-python-v2:$out/qualification"
export GENERATIVEQC_LIBRARY="$out/libgenerativeqc.so.0"
export GENERATIVEQC_RESOURCE_CUDA_TEST=1 GENERATIVEQC_DFT_CUDA_TEST=1
export GENERATIVEQC_SCF_ORACLE_RECEIPTS="$out/oracles"
export LD_LIBRARY_PATH="$out:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_PATH=/group/software/cuda-12.9.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Slurm job: %s; node: %s; assigned CUDA devices: %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
sccache --version
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-before.log"
sha256sum "$out/libgenerativeqc.so.0" "$base_root/artifacts/libgenerativeqc.so.0"
env -i PATH=/nonexistent CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SLURM_JOB_ID="$SLURM_JOB_ID" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$out/sdk-cuda" cuda
printf 'Installed SDK D4 independent reference passed\n'
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -c 'from importlib.metadata import version; print({name: version(name) for name in ("cupy-cuda12x", "gpu4pyscf-cuda12x", "gpu4pyscf-libxc-cuda12x", "dftd4", "pyscf", "numpy")})'
"$out/generativeqc_d4_production_tests" cuda
"$out/generativeqc_d4_ragged_tests" cuda
"$out/generativeqc_d4_eeq_cuda_tests"
"$out/generativeqc_d4_reference_cuda_tests"
"$out/generativeqc_d4_schedule_cuda_tests"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 "$out/generativeqc_d4_production_tests" cuda > "$out/memcheck.log" 2>&1
python -m pytest tests/python/test_named_d4_resources.py tests/python/test_named_ks_resources.py tests/python/test_d3_native_resources.py tests/python/test_d4_production.py tests/python/test_parameterized_d4_composition.py tests/python/test_d3_d4_composition_compatibility.py --junitxml="$out/gpu-tests.xml" -q
for campaign in default budget; do
  if [ "$campaign" = budget ]; then
    export D4_QUALIFY_GLOBAL_BUDGET=1
  else
    export D4_QUALIFY_GLOBAL_BUDGET=0
  fi
  python "$out/qualification/qualify.py" --baseline "$base_root" --candidate "$run_root" --baseline-library "$base_root/artifacts/libgenerativeqc.so.0" --candidate-library "$out/libgenerativeqc.so.0" --backend cuda --groups 8 --output "$out/cuda-$campaign-samples.json.xz"
  for role in baseline candidate; do
    source_root="$run_root"
    library="$out/libgenerativeqc.so.0"
    if [ "$role" = baseline ]; then
      source_root="$base_root"
      library="$base_root/artifacts/libgenerativeqc.so.0"
    fi
    prefix="$out/$campaign-$role-profile"
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true --output="$prefix" python "$out/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$prefix.log" 2>&1
    nsys stats --force-export=true --force-overwrite=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$prefix-stats" "$prefix.nsys-rep"
  done
  python "$out/qualification/profile_summary.py" --baseline "$out/$campaign-baseline-profile.sqlite" --candidate "$out/$campaign-candidate-profile.sqlite" --output "$out/$campaign-profile-summary.json"
done
sha256sum "$out/libgenerativeqc.so.0" "$base_root/artifacts/libgenerativeqc.so.0"
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-after.log"

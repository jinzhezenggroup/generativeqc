#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-cuda-basis-9b29bd65d-v2-20261029
base_root=/home/jzzeng/codes/qc-2227-d4-ledger-baseline-228a2a68c-20261029
reference_root=/home/jzzeng/codes/qc-2227-precision-policy-20261029
out="$run_root/artifacts"
library="$base_root/artifacts/libgenerativeqc.so.0"
cd "$run_root"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH="$run_root/python:$run_root:$reference_root/artifacts/reference-python-v2"
export GENERATIVEQC_LIBRARY="$library"
export GENERATIVEQC_RESOURCE_CUDA_TEST=1 GENERATIVEQC_DFT_CUDA_TEST=1
export LD_LIBRARY_PATH="$base_root/artifacts:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_PATH=/group/software/cuda-12.9.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Slurm job: %s; node: %s; assigned CUDA devices: %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
sccache --version
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-before.log"
python -m pytest tests/python/test_native_basis_shape.py tests/python/test_hf_resources_cuda.py tests/python/test_named_ks_resources.py --junitxml="$out/gpu-tests.xml" -q
python "$out/qualification/qualify.py" --baseline "$base_root" --candidate "$run_root" --baseline-library "$library" --candidate-library "$library" --backend cuda --groups 8 --output "$out/cuda-samples.json.xz"
for role in baseline candidate; do
  source_root="$base_root"
  export BASIS_QUALIFY_NATIVE_COUNTS=0
  if [ "$role" = candidate ]; then
    source_root="$run_root"
    export BASIS_QUALIFY_NATIVE_COUNTS=1
  fi
  prefix="$out/$role-profile"
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true --output="$prefix" python "$out/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$prefix.log" 2>&1
  nsys stats --force-export=true --force-overwrite=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$prefix-stats" "$prefix.nsys-rep"
done
python "$out/qualification/profile_summary.py" --baseline "$out/baseline-profile.sqlite" --candidate "$out/candidate-profile.sqlite" --output "$out/profile-summary.json"
sha256sum -c "$out/input-sha256.txt" > "$out/integrity-after.log"

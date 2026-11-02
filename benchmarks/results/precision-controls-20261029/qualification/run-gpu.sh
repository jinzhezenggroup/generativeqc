#!/usr/bin/env bash
set -euo pipefail
run_root=/home/jzzeng/codes/qc-2227-precision-controls-15d8894c9-20261029
base_root=/home/jzzeng/codes/qc-2227-precision-policy-20261029
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
env -i PATH=/nonexistent CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SLURM_JOB_ID="$SLURM_JOB_ID" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$out/sdk" cuda
printf 'Installed SDK precision-budget policy passed\n'
python -c 'from importlib.metadata import version; print({name: version(name) for name in ("cupy-cuda12x", "gpu4pyscf-cuda12x", "gpu4pyscf-libxc-cuda12x", "dftd4", "pyscf", "numpy")})'
python -m pytest -p native_budget_guard tests/python/test_precision_batch_cuda.py tests/python/test_application_fock_cuda.py --junitxml="$out/gpu-tests.xml" -q
python "$out/qualification/qualify.py" --baseline "$base_root" --candidate "$run_root" --baseline-library "$base_root/artifacts/libgenerativeqc.so.0" --candidate-library "$out/libgenerativeqc.so.0" --backend cuda --groups 8 --output "$out/cuda-samples.json.xz"
for role in baseline candidate; do
  source_root="$run_root"
  library="$out/libgenerativeqc.so.0"
  if [ "$role" = baseline ]; then
    source_root="$base_root"
    library="$base_root/artifacts/libgenerativeqc.so.0"
  fi
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true --output="$out/$role-profile" python "$out/qualification/endpoint_worker.py" --source "$source_root" --library "$library" --backend cuda --once > "$out/$role-profile.log" 2>&1
  nsys stats --force-export=true --force-overwrite=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output "$out/$role-profile-stats" "$out/$role-profile.nsys-rep"
done
sha256sum "$out/libgenerativeqc.so.0" "$base_root/artifacts/libgenerativeqc.so.0"

python "$out/qualification/profile_summary.py" --baseline "$out/baseline-profile.sqlite" --candidate "$out/candidate-profile.sqlite" --output "$out/profile-summary.json"

sha256sum -c "$out/input-sha256.txt" > "$out/integrity-after.log"

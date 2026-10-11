#!/usr/bin/env bash
set -euo pipefail
root=/home/jzzeng/codes/qc-array-api-rhf-d15383b49-20261011
cd "$root"
evidence="$root/.artifacts/qualification/$SLURMD_NODENAME-$SLURM_JOB_ID"
mkdir -p "$evidence"
export PATH=/home/jzzeng/miniconda3/bin:/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/codes/wb97m-20261002/ccache/usr/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="$root/python:$root"
export GENERATIVEQC_LIBRARY="$root/build/cuda-release-sm120/libgenerativeqc.so"
export GENERATIVEQC_PROFILE=off
export GENERATIVEQC_ARRAY_API_CUDA_TEST=1
export GENERATIVEQC_TENSOR_CUDA_TEST=1
export GENERATIVEQC_TENSOR_ARCH=sm_120
export NVCC_CCBIN=/usr/bin/g++-12
export CCACHE_BASEDIR="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'job=%s node=%s visible=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES" | tee "$evidence/scheduler.txt"
ccache --version | tee "$evidence/cache-version.txt"
ccache --show-stats > "$evidence/cache-before-run.txt"
nvcc --version > "$evidence/nvcc-version.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$evidence/device-inventory.csv"
sha256sum "$GENERATIVEQC_LIBRARY" > "$evidence/library-sha256.txt"
sha256sum .artifacts/qualification/source.tar.gz > "$evidence/source-sha256.txt"
python -c 'import sys,numpy,pytest; print(sys.version); print("numpy", numpy.__version__, "pytest", pytest.__version__)' > "$evidence/python-version.txt"
python - <<'PY' | tee "$evidence/native-source-identity.txt"
from generativeqc import _native
library = _native.load_library(device="cuda", device_id=0)
identity = library.generativeqc_get_source_identity().decode()
assert identity == "869e302bd2e1dcb0c82d9b005652581c297a28dc1225d6d5b0957562e80815bb", identity
print(identity)
PY
pytest -q tests/python/test_array_api_rhf_benchmark.py \
  tests/python/test_array_api_native_cuda.py tests/python/test_array_api_molecular.py \
  tests/python/test_tensor_transcendentals_cuda.py \
  --junitxml="$evidence/tests.xml" 2>&1 | tee "$evidence/tests.log"
for molecule in h2 water; do
  python benchmarks/array_api_rhf_endpoint.py --backend cuda --cuda-arch sm_120 \
    --molecule "$molecule" --repeats 10 --output "$evidence/$molecule-endpoint.json" \
    > "$evidence/$molecule-endpoint.log" 2>&1
  python - "$evidence/$molecule-endpoint.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print({key: record[key] for key in (
    "molecule", "backend", "array_api_over_native", "energy_error_hartree",
    "gradient_error_hartree_per_bohr", "matching_iteration_counts",
    "gpu_performance_qualified", "scheduler"
)})
PY
done
ccache --show-stats > "$evidence/cache-after-run.txt"
printf 'RUN COMPLETE: %s\n' "$evidence"

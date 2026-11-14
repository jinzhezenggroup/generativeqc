#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun with a finite allocation}"
: "${CUDA_VISIBLE_DEVICES:?Preserve scheduler-assigned device visibility}"
: "${GFN2_RECEIPT_ROOT:?Set the qualification artifact directory}"
root=$GFN2_RECEIPT_ROOT
source_root=$root/source
original=/data/jzzeng/ncu-xtb-20261005
out=$root/current-cuda
mkdir -p "$out"
trap 'status=$?; printf "{\"exit_code\":%s,\"job\":\"%s\",\"pid\":%s}\n" "$status" "$SLURM_JOB_ID" "$$" > "$out/outcome.json"' EXIT
export PATH=/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:/home/jzzeng/miniconda3/bin:/usr/bin:$PATH
export GENERATIVEQC_LIBRARY="$source_root/build-cuda/libgenerativeqc.so"
export PYTHONPATH="$source_root/python"
export LD_LIBRARY_PATH="$source_root/build-cuda:/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export GFN2_CUDART=/group/software/cuda-12.9.1/lib64/libcudart.so.12
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CCACHE=/home/jzzeng/.local/bin/sccache
export GENERATIVEQC_TEST_GFN2_CUDA=1
python=/home/jzzeng/miniconda3/bin/python
cd "$source_root"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/device.csv"
sha256sum "$GENERATIVEQC_LIBRARY" "$original/inputs/cases.json" "$root/matched-oracle.json" > "$out/inputs.sha256"
sha256sum --quiet -c "$root/cuda-source.sha256"
"$CCACHE" --version > "$out/cache-version.txt"
"$CCACHE" --show-stats > "$out/cache-before.txt"
for property in force energy; do
  "$python" benchmarks/results/gfn2-current-source-20261114/replay.py \
    --inputs "$original/inputs/cases.json" --oracle "$root/matched-oracle.json" \
    --source-root "$source_root" --source-manifest "$root/cuda-source.sha256" \
    --output "$out/public-$property.json" --device cuda --property "$property" \
    --cases water-8 water-32 water-128 > "$out/public-$property.log" 2>&1
done
"$python" -m pytest -q tests/python/test_gfn2_native_oracle.py \
  tests/python/test_gfn2_xtb.py tests/python/test_gfn2_runtime_retention.py \
  tests/python/test_gfn2_runtime_bridge_boundary.py \
  tests/python/test_gfn2_cuda_bootstrap.py \
  tests/python/test_gfn2_cpu_cache_transaction.py \
  tests/python/test_gfn2_orbital_snapshot_transaction.py \
  tests/python/test_gfn2_mixer_receipt_transaction.py \
  tests/python/test_gfn2_density_receipt_transaction.py > "$out/contracts.log" 2>&1
timeout --signal=TERM --kill-after=15s 300 compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 86 \
  --log-file "$out/sanitizer.txt" "$python" -m pytest -q \
  'tests/python/test_gfn2_runtime_retention.py::test_public_results_outlive_reuse_failure_and_cache_clear[cuda]' \
  > "$out/sanitizer-test.log" 2>&1
"$CCACHE" --show-stats > "$out/cache-after.txt"

# The unchanged peer is timed separately and is never the scientific oracle.
export XTBLOOM_LIBRARY="$original/reference/libxtbloom.so"
export PYTHONPATH=/data/jzzeng/cc-1901-20261005/integrated-v3/python:/data/jzzeng/cc-1901-20261005/integrated-v3
export LD_LIBRARY_PATH="$original/reference:$LD_LIBRARY_PATH"
/home/jzzeng/xtb-perf-20261003/env/bin/python "$original/measure_v1.py" \
  --engine xtbloom --property force --cases water-128 \
  --output "$out/xtbloom-force.json" > "$out/xtbloom-force.log" 2>&1

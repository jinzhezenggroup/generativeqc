#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?GPU work requires srun}"
: "${CUDA_VISIBLE_DEVICES:?Preserve Slurm device visibility}"
root=/data/jzzeng/qc-hybrid-default-auto-20261011-validation
source="$root/source"
bin="$root/bin"
deps=/data/jzzeng/issue1972-20261005/deps
envroot=/home/jzzeng/codes/wb97m-20261002
cuda=/group/software/cuda-12.9.1
out="$root/results/$SLURM_JOB_ID"
mkdir -p "$out" "$root/tmp"
trap 'status=$?; printf "{\"exit_code\":%s,\"job\":\"%s\"}\n" "$status" "$SLURM_JOB_ID" > "$out/outcome.json"' EXIT
export PATH="$deps/bin:$cuda/bin:/home/jzzeng/miniconda3/bin:$PATH"
export PYTHONPATH="$source/python:$source:$deps"
export LD_LIBRARY_PATH="$bin:$deps/pyscf/lib:$deps/lib:$envroot/ccache/usr/lib/x86_64-linux-gnu:$envroot/gpu-env/lib/python3.13/site-packages/cutensor/lib:$cuda/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$cuda/lib64/libcudart.so.12:$cuda/lib64/libcublas.so.12:$cuda/lib64/libcublasLt.so.12:$cuda/lib64/libcusolver.so.11:$cuda/lib64/libcusparse.so.12"
export GENERATIVEQC_LIBRARY="$bin/libgenerativeqc.so"
export GENERATIVEQC_KS_PHASE_VALUES_PROBE="$bin/ks_phase_values_probe"
export GENERATIVEQC_DFT_CUDA_TEST=1
export GENERATIVEQC_CCACHE="$deps/bin/ccache"
export CCACHE_DIR=/data/jzzeng/ccache CCACHE_BASEDIR="$source"
export GENERATIVEQC_NVCC="$cuda/bin/nvcc" CUDACXX="$cuda/bin/nvcc" CXX=/usr/bin/g++
export NVCC_CCBIN=/usr/bin/g++-11
export GENERATIVEQC_STATIONARY_CACHE="$root/cache/compiled"
export GENERATIVEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS=8
export TMPDIR="$root/tmp"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python="$envroot/gpu-env/bin/python"
cd "$source"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/gpu.csv"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$out/visibility.txt"
ccache --version > "$out/ccache-version.txt"
sha256sum "$bin"/*so* "$bin/ks_phase_values_probe" "$bin/generativeqc_stationary_pbe0_rks_spd.json" > "$out/binary-SHA256SUMS"
if [[ ${1:-abba} == probes ]]; then
  export GENERATIVEQC_DF_PROGRESS_TRACE="$out/small-trace.jsonl"
  "$python" -m pytest -x -v tests/python/test_ks_phase_resident_values_cuda.py -k water \
    --basetemp="$out/small-gates" > "$out/small-gates.log" 2>&1
  export GENERATIVEQC_RCCSD_CUDA_TEST=1
  "$python" -m pytest -x -v tests/python/test_rhf_phase_resident_values_cuda.py \
    --basetemp="$out/rhf-gates" > "$out/rhf-gates.log" 2>&1
elif [[ ${1:-abba} == memcheck ]]; then
  export GENERATIVEQC_CUDA_KS_CHUNK=1
  export GENERATIVEQC_DF_PROGRESS_TRACE="$out/memcheck-trace.jsonl"
  input=$(find "$root/results" -path '*/small-gates/*' -name water.input -print -quit)
  test -n "$input"
  cp "$input" "$out/water.input"
  compute-sanitizer --tool memcheck --error-exitcode 86 \
    "$bin/ks_phase_values_probe" "$out/water.input" \
    > "$out/memcheck.log" 2>&1
  grep -F 'ERROR SUMMARY: 0 errors' "$out/memcheck.log" > "$out/sanitizer-accepted.txt"
else
  for run in A1 B1 B2 A2; do
    export GENERATIVEQC_KS_RESIDENT_VALUES=0
    [[ $run == B* ]] && unset GENERATIVEQC_KS_RESIDENT_VALUES
    export GENERATIVEQC_DF_PROGRESS_TRACE="$out/trace-$run.jsonl"
    "$python" "$root/observe-default.py" native --atoms 96 \
      --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
      --reference "$root/reference.json" --repeats 1 \
      --output "$out/pbe0-96-$run.json" > "$out/pbe0-96-$run.log" 2>&1
  done
  export GENERATIVEQC_KS_RESIDENT_VALUES=0 EXPECT_SOURCE_COLD=0
  export GENERATIVEQC_DF_PROGRESS_TRACE="$out/reference-small-trace.jsonl"
  "$python" -m benchmarks.readme_pbe0 reference --atoms 3 \
    --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats 1 \
    --output "$out/reference-small.json" > "$out/reference-small.log" 2>&1
  for selection in 0 unset; do
    export GENERATIVEQC_KS_RESIDENT_VALUES=0
    [[ $selection == unset ]] && unset GENERATIVEQC_KS_RESIDENT_VALUES
    export GENERATIVEQC_DF_PROGRESS_TRACE="$out/trace-small-$selection.jsonl"
    "$python" "$root/observe-default.py" native --atoms 3 \
      --basis-file benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json \
      --reference "$out/reference-small.json" --repeats 1 \
      --output "$out/pbe0-3-$selection.json" > "$out/pbe0-3-$selection.log" 2>&1
  done
fi
ccache --show-stats > "$out/ccache-after.txt"
printf '%s\n' complete > "$out/status.txt"

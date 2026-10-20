#!/usr/bin/env bash
set -euo pipefail
source /data/jzzeng/qc-cbs-fast-mixed-20261020/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_PROFILE=off
output="$CBS_FAST_ROOT/fast-diagnosis-20261021"
variant="$1"
carbon="$2"
trace="${3:-0}"
suffix="${4:-}"
policy="${5:-fast}"
if [[ "$variant" == baseline ]]; then
  export GENERATIVEQC_LIBRARY="$CBS_FAST_ROOT/runtime/libgenerativeqc.so"
else
  export GENERATIVEQC_LIBRARY="$output/libgenerativeqc-$variant.so"
fi
name="c$carbon-$variant"
if [[ "$trace" == 1 ]]; then
  name="$name-profile"
  export GENERATIVEQC_DF_PROGRESS_TRACE="$output/$name.trace.jsonl"
  test ! -e "$GENERATIVEQC_DF_PROGRESS_TRACE"
fi
name="$name$suffix"
compat_root=/data/jzzeng/qc-r2scan-force-branches-20261015
export PYTHONPATH="$compat_root/oracle-python:$PYTHONPATH"
"$compat_root/compat/ld-linux-x86-64.so.2" \
  --preload "$GENERATIVEQC_LIBRARY:/group/software/cuda-12.9.1/lib64/libcudart.so.12:/group/software/cuda-12.9.1/lib64/libcublas.so.12:/group/software/cuda-12.9.1/lib64/libcusolver.so.11" \
  --library-path "$compat_root/compat:$compat_root/common-lib:/group/software/cuda-12.9.1/lib64:$LD_LIBRARY_PATH" \
  /home/jzzeng/codes/wb97m-20261002/gpu-env/bin/python \
  -m benchmarks.readme_cbs native \
  --qm7 /data/jzzeng/readme-cbs-20261013/qm7.mat --carbon "$carbon" \
  --policy "$policy" --hf-reference df --budget-mib 32768 --samples 1 \
  --source-revision "6339327feda0e28d961f45f3c6f5944730f1024d+$variant" \
  --output "$output/$name.json"

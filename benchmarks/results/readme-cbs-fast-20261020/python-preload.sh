#!/usr/bin/env bash
set -euo pipefail
compat_root=/data/jzzeng/qc-r2scan-force-branches-20261015
export PYTHONPATH="$compat_root/oracle-python${PYTHONPATH:+:$PYTHONPATH}"
exec "$compat_root/compat/ld-linux-x86-64.so.2" \
  --preload "$CBS_FAST_ROOT/runtime/libgenerativeqc.so:/group/software/cuda-12.9.1/lib64/libcudart.so.12:/group/software/cuda-12.9.1/lib64/libcublas.so.12:/group/software/cuda-12.9.1/lib64/libcusolver.so.11" \
  --library-path "$compat_root/compat:$compat_root/common-lib:/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /home/jzzeng/codes/wb97m-20261002/gpu-env/bin/python "$@"

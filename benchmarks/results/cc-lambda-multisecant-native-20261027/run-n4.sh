#!/usr/bin/env bash
set -euo pipefail
task_root=/data/jzzeng/cc-multisecant-20261027
export PATH=/home/jzzeng/.local/bin:/home/jzzeng/miniconda3/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$task_root/pydeps:$task_root/source/python:$task_root/source"
export GENERATIVEQC_LIBRARY="$task_root/libgenerativeqc.so"
cd "$task_root/source"
scontrol -d show job "$SLURM_JOB_ID" > "$task_root/allocation-$SLURM_JOB_ID.txt"
sha256sum "$GENERATIVEQC_LIBRARY" "$task_root"/runtime/* > "$task_root/runtime-$SLURM_JOB_ID.sha256"
# Run the source-matched binary with its original userspace ABI. The host system
# is untouched; child scheduler/metadata utilities keep their ordinary loader.
qc_probe_library_path="$task_root/runtime:/home/jzzeng/miniconda3/lib:/group/software/cuda-12.9.1/lib64"
"$task_root/runtime/ld-linux-x86-64.so.2" --library-path "$qc_probe_library_path" \
  /home/jzzeng/miniconda3/bin/python -m pytest -q "$task_root/test_probe.py"
"$task_root/runtime/ld-linux-x86-64.so.2" --library-path "$qc_probe_library_path" \
  /home/jzzeng/miniconda3/bin/python "$task_root/experiment.py" "$@"

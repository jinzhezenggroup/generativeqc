#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/qc-3055-full-df-final
srun --immediate=60 --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=24G --time=01:00:00 bash -lc '
set -euo pipefail
cd /home/jzzeng/codes/qc-3055-full-df-final
export PATH=/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH=python:.
export LD_LIBRARY_PATH=$PWD/native/blas:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export GENERATIVEQC_LIBRARY=$PWD/native/libgenerativeqc.so.0.1.0
export GENERATIVEQC_FULL_DF_CUDA_TEST=1
export GENERATIVEQC_FULL_DF_EVIDENCE_DIR=$PWD/evidence
export GENERATIVEQC_VALIDATION_SOURCE_ID="$(cat source-identity.txt)"
export GENERATIVEQC_DF_PROGRESS_TRACE=$PWD/evidence-progress.jsonl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
printf "HOST=%s JOB=%s VISIBLE=%s\n" "$(hostname)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
sha256sum "$GENERATIVEQC_LIBRARY"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv
/home/jzzeng/miniconda3/bin/python -m pytest -q -s tests/python/test_full_df_cbs_native.py -k "aug-cc-pvqz or augmented_tq or low_angular"
'

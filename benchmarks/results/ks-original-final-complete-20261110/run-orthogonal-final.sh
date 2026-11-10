#!/usr/bin/env bash
set -uo pipefail
cd "${GENERATIVEQC_QUALIFICATION_ROOT:-/home/jzzeng/codes/qc-ks-orthogonal-final-20261110}"
export PATH=/home/jzzeng/.local/bin:/group/software/cuda-12.9.1/bin:$PATH
export PYTHONPATH=python:.:tests/python:$PWD/.reference-deps
export LD_LIBRARY_PATH=$PWD:$PWD/blas:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GENERATIVEQC_KS_STATE_CUDA_TEST=1 GENERATIVEQC_GRID_CUDA_TEST=1
export GENERATIVEQC_LIBRARY=$PWD/candidate-final.so
unset GENERATIVEQC_DF_PROGRESS_TRACE
mkdir -p full-original-gates
printf 'HOST=%s JOB=%s STEP=%s VISIBLE=%s ARM=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_STEP_ID:-}" "$CUDA_VISIBLE_DEVICES" "$1"
cat source-revision.txt
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv
sha256sum "$GENERATIVEQC_LIBRARY" src/dft/{cuda_ks.cpp,cuda_ks_diis_policy.hpp} python/generativeqc/{ks.py,calculator.py,_ks_snapshot.py,_stationary_cuda.py}
sccache --version
sccache --show-stats
if [[ "$1" == endpoints ]]; then
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/rsh-final
    timeout 1800 /home/jzzeng/miniconda3/bin/python -u -m pytest -q tests/python/test_ks_original_rsh_admission_cuda.py --junitxml=full-original-gates/rsh-final.xml
    echo "original_rsh_exit=$? $(date -Iseconds)"
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/uks-final
    timeout 1800 /home/jzzeng/miniconda3/bin/python -u -m pytest -q tests/python/test_ks_scientific_gates_cuda.py --junitxml=full-original-gates/uks-final.xml
    echo "original_uks_exit=$? $(date -Iseconds)"
    timeout 600 ./generativeqc_ks_cuda_tests --df-range-exchange
    echo "native_df_lr_exit=$? $(date -Iseconds)"
else
    export GENERATIVEQC_KS_STATE_EVIDENCE_DIR=$PWD/full-original-gates/water-final-$1
    timeout 7000 /home/jzzeng/miniconda3/bin/python -u -m pytest -q -s tests/python/test_ks_seed_admission_cuda.py -k "original_grid and $1" --junitxml=full-original-gates/water-final-$1.xml
    echo "original_water_${1}_exit=$? $(date -Iseconds)"
fi
sccache --show-stats

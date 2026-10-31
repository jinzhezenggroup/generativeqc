#!/usr/bin/env bash
set -euo pipefail
root=/home/jzzeng/tmp/qc-1123-shared-aot-20261027
base=/home/jzzeng/tmp/qc-1123-baseline-20261027
cd "$root"
export PATH="$root/.venv/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin"
export PYTHONPATH="$root/python:$root"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64"
export GENERATIVEQC_LIBRARY="$root/evidence/full-candidate-e599.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
export PACKAGE_SCOPE=full
REFERENCE_FORCES=evidence/parent-references REBOUND_OUTPUT=evidence/rebound-full python -u rebind-forces.py > evidence/accepted-full-parity.log 2>&1
python -u alias-forces.py > evidence/accepted-alias-forces.log 2>&1
python -u installed-cli.py > evidence/accepted-full-installed.log 2>&1
python -u phased-forces.py evidence/phased-candidate.json > evidence/phased-candidate.log 2>&1
GENERATIVEQC_LIBRARY="$base/evidence/full-incumbent-e599.so" python -u phased-forces.py evidence/phased-baseline.json > evidence/phased-baseline.log 2>&1
GENERATIVEQC_LIBRARY="$base/evidence/full-incumbent-e599.so" REFERENCE_FORCES=evidence/alias-forces REBOUND_OUTPUT=evidence/alias-incumbent python -u rebind-forces.py > evidence/alias-incumbent.log 2>&1

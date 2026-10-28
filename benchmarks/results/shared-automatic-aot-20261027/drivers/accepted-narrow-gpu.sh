#!/usr/bin/env bash
set -euo pipefail
root=/home/jzzeng/tmp/qc-1123-shared-aot-20261027
cd "$root"
export PATH="$root/.venv/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin"
export PYTHONPATH="$root/python:$root"
export LD_LIBRARY_PATH="$root/evidence/narrow-snapshot/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export GENERATIVEQC_LIBRARY="$root/evidence/narrow-candidate-e599.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1
export GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR="$root/evidence/accepted-extra"
export GENERATIVEQC_AUTOMATIC_CUDA_XC_EXECUTABLE="$root/evidence/narrow-snapshot/build/pro6000-cuda/generativeqc_dft_cuda_tests"
mkdir -p evidence/accepted-extra
./evidence/narrow-snapshot/build/pro6000-cuda/generativeqc_dft_cuda_tests > evidence/accepted-dft-native.log 2>&1
./evidence/narrow-snapshot/build/pro6000-cuda/generativeqc_ks_cuda_tests > evidence/accepted-ks-native.log 2>&1
python -m pytest -x -q tests/python/test_libxc_automatic_cuda_xc.py > evidence/accepted-points.log 2>&1
GENERATIVEQC_AUTOMATIC_CUDA_KS_TEST=1 python -m pytest -x -q tests/python/test_libxc_automatic_cuda_ks.py > evidence/accepted-ks.log 2>&1
python -m pytest -x -q tests/python/test_libxc_native_cuda_forces.py -k 'minimum_and_below_budget or ragged_failure or nonconvergence' > evidence/accepted-failures.log 2>&1
REFERENCE_FORCES=evidence/parent-references REBOUND_OUTPUT=evidence/rebound-final python -u rebind-forces.py > evidence/accepted-parity.log 2>&1
python -u installed-cli.py > evidence/accepted-installed.log 2>&1
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 python -u sanitizer-force.py > evidence/accepted-force-memcheck.log 2>&1
compute-sanitizer --tool initcheck --error-exitcode 99 python -u sanitizer-force.py > evidence/accepted-force-initcheck.log 2>&1
compute-sanitizer --tool initcheck --target-processes all --log-file evidence/staged-initcheck-%p.log --error-exitcode 99 python -m pytest -x -q tests/python/test_libxc_automatic_cuda_xc.py -k staged > evidence/accepted-point-initcheck.log 2>&1

#!/bin/sh
# Explicit small-CPU qualification of this checkout using its own native library.
set -eu
: "${GENERATIVEQC_LIBRARY:?Set this checkout's own CPU-only native library}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export GENERATIVEQC_PROFILE=off PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/python:$PWD"
CPU=${CPU:-2}
python - <<'PY'
import ctypes
from pathlib import Path
from generativeqc import _native
from generativeqc.autotune import source_identity
library = _native.load_library()
library.generativeqc_get_source_identity.restype = ctypes.c_char_p
assert library.generativeqc_get_source_identity().decode() == source_identity(Path.cwd()), "Mixed source/library identity"
PY
taskset -c "$CPU" ctest --test-dir build/cpu-revalidation --output-on-failure -j1 \
  -R '^generativeqc_(dft|dft_density_source|uks|uks_state|ks_final_state|final_state|rks_response|uks_response|scf_diagnostic|self_consistent|native|uhf|scf_proposal)_tests$'
taskset -c "$CPU" python -m pytest -p no:cacheprovider -q -rs \
  tests/python/test_cpu_uks_primary_eligibility.py \
  tests/python/test_dft_scf.py tests/python/test_dft_batch.py \
  tests/python/test_ks_resources.py tests/python/test_ks_mixed_resources.py -k 'not cuda'
# The unchanged publication-union controls cover additional semilocal/hybrid
# and compiler policies; real-device/transport-only skips remain explicit.
sh benchmarks/results/cpu-ks-final-closure-20261003/run_union_controls.sh

# Existing small-system response/nonlocal/RSH/WB and PBE-UKS force controls.
taskset -c "$CPU" python -m pytest -p no:cacheprovider -q -rs \
  tests/python/test_response_native_rks.py tests/python/test_response_native_uks.py \
  tests/python/test_vv10_self_consistent.py tests/python/test_stationary_rsh_cpu.py \
  tests/python/test_wb97mv_complete.py::test_public_wb97mv_live_state_and_complete_gradient \
  'tests/python/test_dft_complete_cpu.py::test_complete_open_shell_uks_analytic_and_reconverged_fd[pbe-uks-native]' \
  -k 'not cuda'

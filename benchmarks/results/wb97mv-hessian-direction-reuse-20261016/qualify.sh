#!/usr/bin/env bash
set -euo pipefail
printf 'source=5aa703721fdce6a220838676af615d574f0fed05 job=%s step=%s visibility=%s node=%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "${CUDA_VISIBLE_DEVICES-}" "$(hostname)"
sha256sum generativeqc libgenerativeqc.so.0.1.0 generativeqc_cuda_wb97mv_nuclear_jk_probe generativeqc_ks_wb97mv_hessian_probe inventory.json build_identity.hpp
mkdir -p ".artifacts/slurm-$SLURM_JOB_ID"
export GENERATIVEQC_STATIONARY_RECEIPTS="$PWD/.artifacts/slurm-$SLURM_JOB_ID/receipts"
python=/home/jzzeng/codes/wb97m-20261002/gpu-env/bin/python
./generativeqc_cuda_wb97mv_jk_closure_tests
"$python" -m pytest -xq -rs --durations=10 tests/python/test_native_wb97mv_nuclear_jk.py
"$python" -m pytest -xq -rs --durations=10 tests/python/test_native_wb97mv_stationary.py -k resident_and_streamed
"$python" -m pytest -xq -rs --durations=10 tests/python/test_wb97mv_hessian_cli_cuda.py
"$python" -m pytest -xq -rs --durations=10 \
  'tests/python/test_native_wb97mv_stationary.py::test_high_angular_molecular_stationary_hvp[rks-direct-f-cartesian]' \
  'tests/python/test_native_wb97mv_stationary.py::test_high_angular_molecular_stationary_hvp[uks-df-ecp-f-g-cartesian]' \
  'tests/python/test_native_wb97mv_stationary.py::test_stationary_hvp_against_reconverged_independent_energies[True-True-True-1]'
printf 'FINAL_QUALIFICATION_PASSED\n'

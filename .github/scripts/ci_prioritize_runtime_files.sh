#!/usr/bin/env bash
# Order only: preserve every selected file, duplicate, and missing-file failure.
ci_prioritize_runtime_files() {
  # PR #2888/run 41152461345 timed out at 30 minutes. The four-case ECP
  # Hessian file started after minute 25, and its final case was still running
  # after all other files completed. Start it first, followed by descending
  # observed whole-file elapsed time; preserve all module-local fixtures.
  local -a priority=(
    tests/python/test_wb97mv_ecp_hessian.py
    tests/python/test_uccsd_triples_response.py
    tests/python/test_native_stationary_phased_budget.py
    tests/python/test_correlation_convergence.py
    tests/python/test_bulk_aot_clangxx_closure.py
    tests/python/test_ccsd_t_orbital_response.py
    tests/python/test_cc_lambda_response.py
    tests/python/test_cc_lambda_solver.py
    tests/python/test_codegen_schedules.py
    tests/python/test_df_lambda_matrix.py
    tests/python/test_cc_solver.py
    tests/python/test_df_cc_native_solver.py
    tests/python/test_uccsd_solver.py
    tests/python/test_cc_triples_response_cuda.py
    tests/python/test_cpu_hf_high_l.py
    tests/python/test_df_cc_mixed_precision.py
    tests/python/test_ecp_multicenter.py
    tests/python/test_cc_triples_lambda_response.py
    tests/python/test_df_rys.py
    tests/python/test_df_lambda_reduction.py
    tests/python/test_response_native_uks.py
    tests/python/test_one_electron_cpu_codegen.py
    tests/python/test_mp2_public.py
    tests/python/test_grid_policy_convergence.py
    tests/python/test_implicit_response.py
    tests/python/test_hessian_numerical.py
    tests/python/test_hessian_block.py
    tests/python/test_dft_batch.py
    tests/python/test_cuda_xc_point_contract.py
    tests/python/test_response_native_rks.py
  )
  local -a ordered=()
  local -A ranked=()
  local path target
  for path in "${priority[@]}"; do
    ranked["$path"]=1
    for target in "${test_targets[@]}"; do
      if [[ "$target" == "$path" ]]; then
        ordered+=("$target")
      fi
    done
  done
  for target in "${test_targets[@]}"; do
    if [[ -z "${ranked[$target]:-}" ]]; then
      ordered+=("$target")
    fi
  done
  test_targets=("${ordered[@]}")
}
ci_prioritize_runtime_files
unset -f ci_prioritize_runtime_files

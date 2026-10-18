#!/usr/bin/env bash
# Source with test_targets already selected. This is an order-only transform:
# retain missing, duplicate, and unrecognized targets for pytest to handle.
ci_prioritize_posthf_files() {
  # Descending whole-file worker seconds from PR #2386/run 38306920093,
  # corroborated by PR #2387/run 38308786441. Counts alone underweight the
  # 10-case physical-response file and leave several workers idle at the end.
  local -a priority=(
    tests/python/test_uccsd_lambda.py
    tests/python/test_uccsd_t_owner.py
    tests/python/test_uccsd_physical_response.py
    tests/python/test_dft_complete_cpu.py
    tests/python/test_hessian_directional.py
    tests/python/test_cc_complete_gradient.py
    # Complete CBS module measured 309-329 s; retain its shared fixtures.
    tests/python/test_ccsdt_cbs_native.py
    tests/python/test_ecp_heavy.py
    tests/python/test_ccsd_t_complete_gradient.py
    tests/python/test_ecp_stationary_cpu.py
    tests/python/test_cc_triples_response.py
    tests/python/test_ecp_f.py
    tests/python/test_hessian_reference.py
    tests/python/test_hessian_analytic.py
    tests/python/test_response_operator.py
    tests/python/test_hessian_hvp.py
    tests/python/test_ecp_f_projector.py
    tests/python/test_ccsd_t_gradient_validation.py
    tests/python/test_mp2_gradient.py
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
  # New/unmeasured targets keep their original relative order and multiplicity.
  # Never add an absent priority target or drop an unfamiliar selected target.
  for target in "${test_targets[@]}"; do
    if [[ -z "${ranked[$target]:-}" ]]; then
      ordered+=("$target")
    fi
  done
  test_targets=("${ordered[@]}")
}
ci_prioritize_posthf_files
unset -f ci_prioritize_posthf_files

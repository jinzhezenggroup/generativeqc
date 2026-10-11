#!/usr/bin/env bash
# Source from the CI Python test step (or directly from shell regressions).
# Required environment: GITHUB_EVENT_NAME, GENERATIVEQC_CI_SHARD.
# Outputs: test_targets, dist_mode, coverage_args, extra_args.
# Keeping this selection out of workflow YAML avoids GitHub's 21,000-character
# run-step limit without disabling or silently splitting scientific tests.
: "${GITHUB_EVENT_NAME:?}"
: "${GENERATIVEQC_CI_SHARD:?}"

extra_args=()
coverage_args=()
if [[ "${GITHUB_EVENT_NAME}" != "merge_group" ]]; then
  coverage_args+=(--cov --cov-report="xml:coverage-python-${GENERATIVEQC_CI_SHARD}.xml")
fi
if [[ "${GITHUB_EVENT_NAME}" != "schedule" && "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  # Keep exhaustive qualification matrices on scheduled/manual CI.
  # Routine PR/master CI retains one complete CPU HF endpoint for
  # orbital-f and f-projector support, while avoiding redundant
  # spin/representation variants that dominate wall time under
  # --dist=loadfile.
  extra_args+=(
    --deselect
    'tests/python/test_mp2_gradient.py::test_dense_complete_gradient_matches_fully_resolved_finite_differences[water]'
    # The complete 55-class capability report performs bounded source
    # emission for every recurrence/schedule combination and costs
    # several minutes by itself. Focused capability/emitter tests stay
    # routine; retain the exhaustive catalog sweep nightly/manual.
    --deselect
    'tests/python/test_codegen_production.py::test_codegen_capability_report_covers_catalog_and_manifest'
    --deselect
    'tests/python/test_ecp_f.py::test_f_prepared_replay_with_planned_budget[cpu]'
    --deselect
    'tests/python/test_ecp_f.py::test_f_complete_hf_energy_and_force[1-cartesian-cpu]'
    --deselect
    'tests/python/test_ecp_f.py::test_f_complete_hf_energy_and_force[0-spherical-cpu]'
    --deselect
    'tests/python/test_ecp_f.py::test_f_complete_hf_energy_and_force[1-spherical-cpu]'
    --deselect
    'tests/python/test_ecp_f_projector.py::test_f_projector_complete_hf[1-cartesian-cpu]'
    --deselect
    'tests/python/test_ecp_f_projector.py::test_f_projector_complete_hf[0-spherical-cpu]'
    --deselect
    'tests/python/test_ecp_f_projector.py::test_f_projector_complete_hf[1-spherical-cpu]'
    # Keep one direct f-shell complete-HF endpoint on routine CI.
    # The matching full projector replay is independently covered by
    # lighter projector contracts here and by the complete nightly run.
    --deselect
    'tests/python/test_ecp_f_projector.py::test_f_projector_complete_hf[0-cartesian-cpu]'
    # Correlated-gradient acceptance contains reconverged molecular
    # finite differences and larger-molecule independent references.
    # H2/small-contract coverage stays routine; broad endpoint replay
    # belongs to scheduled/manual qualification.
    # The NH3 complete RCCSD(T) analytic-gradient endpoint is a
    # separate ~4.7 minute native solve on hosted runners. H2O stays
    # routine and is shared by the component/response checks through
    # the module cache; keep NH3 for scheduled/manual qualification.
    --deselect
    'tests/python/test_ccsd_t_complete_gradient.py::test_complete_ccsdt_gradient_matches_pinned_pyscf[nh3]'
    --deselect
    'tests/python/test_cc_complete_gradient.py::test_complete_native_endpoint_matches_pinned_pyscf_gradient[ch4]'
    --deselect
    'tests/python/test_cc_complete_gradient.py::test_complete_native_endpoint_matches_pinned_pyscf_gradient[nh3]'
    --deselect
    'tests/python/test_cc_complete_gradient.py::test_nuclear_finite_differences_resolve_hf_and_cc_at_three_steps[nh3]'
    --deselect
    'tests/python/test_cc_complete_gradient.py::test_nuclear_finite_differences_resolve_hf_and_cc_at_three_steps[h2o]'
    --deselect
    'tests/python/test_cc_complete_gradient.py::test_nuclear_finite_differences_resolve_hf_and_cc_at_three_steps[h2_d_spherical]'
    # These are qualification-scale replays (roughly 1-2 minutes
    # each on hosted runners); smaller MP2/DFT/ECP/HVP contracts still
    # execute on every PR.
    --deselect
    'tests/python/test_mp2_gradient_fast.py::test_water_complete_gradient_matches_directional_finite_differences'
    --deselect
    'tests/python/test_ecp_heavy.py::test_heavy_complete_hf_and_core_bookkeeping[cpu-1-Au]'
    --deselect
    'tests/python/test_dft_complete_cpu.py::test_complete_asymmetric_water_analytic_and_reconverged_fd[lda-rks-reference]'
    --deselect
    'tests/python/test_second_derivatives_inputs.py::test_public_hvp_arbitrary_rotation_and_bra_shell_permutation[cpu-eri]'
    --deselect
    'tests/python/test_second_derivatives_inputs.py::test_contracted_public_hvp_matches_three_step_independent_gradient_fd[cpu-False-eri]'
    # Retain one representative high-angular/range path routinely;
    # move exhaustive symbolic/long-range qualification to nightly.
    --deselect
    'tests/python/test_high_angular.py::test_g_one_electron_all_components_libcint[nuclear_attraction-angular3]'
    --deselect
    'tests/python/test_codegen_high_l.py::test_dddp_rys5_recurrence_matches_every_symbolic_component'
    --deselect
    'tests/python/test_range_separation.py::test_generated_range_values_and_all_center_derivatives_against_libcint[angular3-cpu-long_range]'
    # WB97M-V reconverged finite-difference replays are qualification
    # scale on hosted runners. The coarse-grid water/OH cases take
    # ~9.5/~12.8 minutes, and the production-grid water RKS case took
    # 772 s in PR #1225. Keep the faster production-grid UKS endpoint
    # on routine CI and run these long replays scheduled/manual.
    --deselect
    'tests/python/test_wb97mv_complete.py::test_public_wb97mv_force_matches_reconverged_energy_differences[grid_shape0-water-rks]'
    --deselect
    'tests/python/test_wb97mv_complete.py::test_public_wb97mv_force_matches_reconverged_energy_differences[grid_shape0-oh-doublet-uks]'
    --deselect
    'tests/python/test_wb97mv_complete.py::test_public_wb97mv_force_matches_reconverged_energy_differences[grid_shape1-water-rks]'
    # The remaining production-grid OH UKS replay is still a full
    # seven-point reconverged finite-difference qualification. Hosted
    # runners measured it at 355-508 s by itself, making one test the
    # routine core critical path. Public WB97M-V live-state/analytic
    # gradient, semilocal RKS/UKS, Maple/Libxc, and generated-code
    # contracts remain on every PR; keep this full FD replay nightly.
    --deselect
    'tests/python/test_wb97mv_complete.py::test_public_wb97mv_force_matches_reconverged_energy_differences[grid_shape1-oh-doublet-uks]'
    # The module-scoped analytic Hessian fixture computes a complete
    # water Hessian before the first parametrized check. On hosted
    # runners that setup alone is ~27 minutes, so keep the H2 path
    # on routine CI and run the water qualification nightly/manual.
    --deselect
    'tests/python/test_hessian_analytic.py::test_first_order_sources_match_independent_native_derivatives[water]'
    --deselect
    'tests/python/test_hessian_analytic.py::test_reference_is_the_native_converged_state[water]'
    --deselect
    'tests/python/test_hessian_analytic.py::test_total_matches_independent_pyscf_hessian[water]'
    --deselect
    'tests/python/test_hessian_analytic.py::test_components_match_independent_finite_difference_oracle[water]'
    --deselect
    'tests/python/test_hessian_analytic.py::test_raw_total_invariants[water]'
    --deselect
    'tests/python/test_hessian_analytic.py::test_three_step_directional_differences_of_native_forces[water]'
    # Full >12-AO RKS HVP qualifications compile and evaluate the
    # complete second-order path and are too expensive for routine
    # hosted feedback. Keep the lightweight synthetic >12-AO/>4-atom
    # admission/resource contract on every PR, and run both native
    # large-domain numerical qualifications scheduled/manual.
    --deselect
    'tests/python/test_hessian_rks_directional.py::test_rks_hvp_admits_resource_bounded_domain_above_12_aos'
    --deselect
    'tests/python/test_hessian_rks_larger_multicenter.py::test_larger_multicenter_rks_hvp_matches_nonzero_gradient_difference'
    # Au resource-boundary replay is a full qualification loop
    # (multiple converged displaced calculations) and currently
    # costs another ~7-10 minutes per spin case. The ordinary Au
    # energy/force checks remain on every PR.
    --deselect
    'tests/python/test_ecp_heavy.py::test_heavy_resource_boundary_replay_complete_energy_difference[cpu-0-Au]'
    --deselect
    'tests/python/test_ecp_heavy.py::test_heavy_resource_boundary_replay_complete_energy_difference[cpu-1-Au]'
    # Keep one response-functional variant on routine CI; the
    # alternate LDA reconverged response remains in nightly/manual.
    --deselect
    'tests/python/test_response_native_rks.py::test_complete_solve_and_reconverged_density_response[lda-rks]'
    --deselect
    'tests/python/test_response_native_uks.py::test_solve_reconverged_spin_densities_and_recycling[lda-uks]'
    # Standard production-grid RKS/UKS endpoints remain routine.
    # The tighter-profile duplicates are qualification coverage.
    --deselect
    'tests/python/test_dft_complete_cpu.py::test_production_grid_converges_against_independent_dense_quadrature[lda-rks-tight]'
    --deselect
    'tests/python/test_dft_complete_cpu.py::test_production_grid_converges_against_independent_dense_quadrature[pbe-rks-tight]'
    --deselect
    'tests/python/test_dft_complete_cpu.py::test_production_grid_open_shell_converges_against_independent_dense_quadrature[lda-uks-tight]'
    --deselect
    'tests/python/test_dft_complete_cpu.py::test_production_grid_open_shell_converges_against_independent_dense_quadrature[pbe-uks-tight]'
    # Closed-shell SPD ECP reconverged finite differences are
    # qualification-scale on hosted runners. The two LDA-UKS
    # reconverged finite-difference cases are also ~9 minutes each.
    # Keep the much faster PBE-UKS complete-force checks plus paired
    # and budgeted ECP contracts on routine CI; run the redundant
    # RKS/LDA qualification matrix on scheduled/manual CI.
    --deselect
    'tests/python/test_ecp_spd_cartesian_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-rks]'
    --deselect
    'tests/python/test_ecp_spd_spherical_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-rks]'
    --deselect
    'tests/python/test_ecp_spd_cartesian_cpu.py::test_spd_force_analytic_and_reconverged_fd[pbe-rks]'
    --deselect
    'tests/python/test_ecp_spd_spherical_cpu.py::test_spd_force_analytic_and_reconverged_fd[pbe-rks]'
    --deselect
    'tests/python/test_ecp_spd_cartesian_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-uks]'
    --deselect
    'tests/python/test_ecp_spd_spherical_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-uks]'
  )
fi
# Keep every routine Python test on every PR, but remove the single
# 26-minute critical path by separating the known numerical/codegen
# tails. New test files land in core automatically until they become
# expensive enough to earn an explicit shard.
dist_mode=loadfile
slow_paths=(
  tests/python/test_cc_complete_gradient.py
  tests/python/test_cc_triples_response.py
  tests/python/test_ccsd_t_gradient_validation.py
  tests/python/test_ccsd_t_complete_gradient.py
  tests/python/test_dft_complete_cpu.py
  tests/python/test_ecp_stationary_cpu.py
  tests/python/test_ecp_public_cpu.py
  tests/python/test_ecp_spd_cartesian_cpu.py
  tests/python/test_ecp_spd_spherical_cpu.py
  tests/python/test_ecp_stuttgart_dft.py
  tests/python/test_ecp_f.py
  tests/python/test_ecp_f_projector.py
  tests/python/test_ecp_heavy.py
  tests/python/test_mp2_gradient.py
  tests/python/test_mp2_gradient_fast.py
  tests/python/test_hessian_analytic.py
  tests/python/test_hessian_directional.py
  tests/python/test_response_operator.py
  tests/python/test_codegen_high_l.py
  tests/python/test_codegen_production.py
  tests/python/test_codegen_cuda_compile.py
  tests/python/test_codegen_autotune.py
  tests/python/test_codegen_profiles.py
  tests/python/test_d3_geometry_ir.py
  tests/python/test_d3_generated_ragged.py
  tests/python/test_high_angular.py
  tests/python/test_one_electron_derivatives.py
  tests/python/test_range_separation.py
  tests/python/test_second_derivatives_inputs.py
  # Empirically expensive routine files live in their own shard so
  # they cannot silently expand the broad core critical path.
  tests/python/test_response_native_rks.py
  tests/python/test_response_native_uks.py
  tests/python/test_mp2_public.py
  tests/python/test_ccsd_t_orbital_response.py
  tests/python/test_hessian_block.py
  tests/python/test_grid_policy_convergence.py
  tests/python/test_cc_triples_response_cuda.py
  tests/python/test_codegen_schedules.py
  tests/python/test_dft_batch.py
  tests/python/test_ecp_multicenter.py
  tests/python/test_cc_triples_lambda_response.py
  tests/python/test_cc_lambda_response.py
  tests/python/test_implicit_response.py
  tests/python/test_df_rys.py
  tests/python/test_hessian_numerical.py
  tests/python/test_one_electron_cpu_codegen.py
  tests/python/test_cc_solver.py
  tests/python/test_cc_lambda_solver.py
  # These compiler/code-generation audits dominated core-a on the
  # 2026-10-04 routine PR run (~347s derivative AOT audit, ~199s
  # deterministic ERI CPU emission, plus native CC codegen). Keep
  # them off the broad core parity split and co-locate them with the
  # existing compiler-heavy shard.
  tests/python/test_derivative_aot_audit.py
  tests/python/test_eri_cpu_codegen.py
  tests/python/test_rccsdt_native_codegen_dependencies.py
  tests/python/test_df_cc_native_codegen.py
)
# PR #1934 / run 37238512811 still left the broad core shards with
# a large measured long tail. Keep these newly measured files out of
# worksteal, but count them in the existing core parity before
# excluding them so unrelated core files do not swap owners.
rebalance_paths=(
  tests/python/test_df_cc_native_solver.py
  tests/python/test_df_lambda_matrix.py
  tests/python/test_df_lambda_reduction.py
  tests/python/test_xc_retirement.py
  tests/python/test_stationary_cooperative_kernel_host.py
  tests/python/test_hessian_reference.py
  tests/python/test_hessian_hvp.py
  tests/python/test_ecp_halogen_dft.py
  # PR #2163 hit the routine cap twice. These compiler/native probes
  # accounted for ~898 worker-seconds in run 37954793838. Transfer
  # whole files to loadfile without changing unrelated core parity.
  tests/python/test_native_contraction_binding.py
  tests/python/test_rccsd_arena_liveness.py
  tests/python/test_cc_cuda_scalar_reduction_codegen.py
  tests/python/test_rccsd_response_admission.py
  tests/python/test_rccsd_iteration_reuse.py
  tests/python/test_stationary_cuda_lowering.py
  tests/python/test_rccsdt_native_triples_response_owner.py
  tests/python/test_becke_normalize_cooperative.py
  tests/python/test_rccsd_numeric_capacity.py
  tests/python/test_stationary_aot_generation.py
  tests/python/test_cc_shared_iteration_driver.py
  tests/python/test_rccsd_codegen_no_numpy.py
  tests/python/test_dft_mp_v1_capacity.py
  tests/python/test_df_lambda_frontier.py
  # Measured full Libxc force lifecycles took ~108 s each in run
  # 38113859405. Count this file in core parity before transferring
  # it, so no unrelated file changes owners as the long tail moves.
  tests/python/test_libxc_automatic_forces.py
  # The complete mixed-RSH force module added ~916-1016 worker-seconds
  # in runs 38121450095/38121385355. Keep its core parity slot while
  # moving the whole matrix to the existing force workers.
  tests/python/test_mixed_rsh_df_forces.py
)
if [[ "$GITHUB_EVENT_NAME" == "schedule" || "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]; then
  # Preserve every other core file's parity while moving WB97M-V.
  rebalance_paths+=(tests/python/test_wb97mv_complete.py)
fi
# Full-nightly Hessian work belongs with the existing file-locality shard,
# not alongside all complete correlated gradients in posthf.
nightly_hessian_paths=(
  tests/python/test_hessian_analytic.py
  tests/python/test_hessian_directional.py
  tests/python/test_hessian_reference.py
  tests/python/test_hessian_hvp.py
)
# The 2026-10-11 PR runs 38123715559, 38122305131, 38122713965,
# and 38121450095 identify a repeatable expensive core-file tail.
# Spread the nine costliest modules into distinct owner groups. These
# originally totaled about 0.7k observed worker-seconds per group. Later
# dedicated-shard transfers take precedence over these historical core pins.
# Keep all scientific tests and all existing nightly-only qualifications.
core_hotspots_a=(
  tests/python/test_mixed_rsh_df_forces.py
)
core_hotspots_b=(
  tests/python/test_df_lambda_core_reuse.py
  tests/python/test_df_lambda_frontier.py
  tests/python/test_df_triples_distinct_moments.py
  tests/python/test_df_lambda_matrix_audit.py
)
core_hotspots_c=(
  tests/python/test_cc_native_tensor_cpu.py
  tests/python/test_cpu_hf_high_l.py
  tests/python/test_cc_canonical_denominators.py
  tests/python/test_direct_order5_sources.py
)
case "${GENERATIVEQC_CI_SHARD}" in
  core-a|core-b|core-c)
    # Whole-file three-way routing keeps module setup ownership intact.
    # Count transferred/nightly-only files before excluding them, so an
    # unrelated test never changes its owner between routine and nightly.
    dist_mode=worksteal
    declare -A slow_lookup=()
    for path in "${slow_paths[@]}"; do
      slow_lookup["${path}"]=1
    done
    declare -A rebalance_lookup=()
    for path in "${rebalance_paths[@]}"; do
      rebalance_lookup["${path}"]=1
    done
    declare -A pinned_core_owner=()
    for path in "${core_hotspots_a[@]}"; do
      pinned_core_owner["${path}"]=0
    done
    for path in "${core_hotspots_b[@]}"; do
      pinned_core_owner["${path}"]=1
    done
    for path in "${core_hotspots_c[@]}"; do
      pinned_core_owner["${path}"]=2
    done
    case "${GENERATIVEQC_CI_SHARD}" in
      core-a) target_bucket=0 ;;
      core-b) target_bucket=1 ;;
      core-c) target_bucket=2 ;;
    esac
    test_targets=()
    core_index=0
    while IFS= read -r path; do
      if [[ -n "${slow_lookup[${path}]:-}" ]]; then
        continue
      fi
      current_index="${core_index}"
      ((core_index += 1))
      if [[ -n "${rebalance_lookup[${path}]:-}" ]]; then
        continue
      fi
      owner=$(( current_index % 3 ))
      if [[ -n "${pinned_core_owner[${path}]:-}" ]]; then
        owner="${pinned_core_owner[${path}]}"
      fi
      if (( owner == target_bucket )); then
        test_targets+=("${path}")
      fi
    done < <(git ls-files tests/python | grep -E '(^|/)test_[^/]*\.py$' | sort)
    if (( ${#test_targets[@]} == 0 )); then
      echo "::error::core shard selected no tests"
      exit 2
    fi
    ;;
  runtime-heavy)
    # Keep file-scoped/module-scoped expensive fixtures on one worker.
    # worksteal can duplicate those setups across workers and inflate
    # wall time even when the collected test count barely changes.
    dist_mode=loadfile
    test_targets=(
      tests/python/test_response_native_rks.py
      tests/python/test_response_native_uks.py
      tests/python/test_mp2_public.py
      tests/python/test_ccsd_t_orbital_response.py
      tests/python/test_hessian_block.py
      tests/python/test_grid_policy_convergence.py
      tests/python/test_cc_triples_response_cuda.py
      tests/python/test_codegen_schedules.py
      tests/python/test_dft_batch.py
      tests/python/test_ecp_multicenter.py
      tests/python/test_cc_triples_lambda_response.py
      tests/python/test_cc_lambda_response.py
      tests/python/test_implicit_response.py
      tests/python/test_df_rys.py
      tests/python/test_hessian_numerical.py
      tests/python/test_one_electron_cpu_codegen.py
      tests/python/test_cc_solver.py
      tests/python/test_cc_lambda_solver.py
      # Keep the module-scoped native solver probe on one worker:
      # worksteal repeated its ~131-160 s setup across four workers.
      tests/python/test_df_cc_native_solver.py
      tests/python/test_df_lambda_matrix.py
      tests/python/test_df_lambda_reduction.py
    )
    ;;
  posthf)
    test_targets=(
      tests/python/test_cc_complete_gradient.py
      tests/python/test_cc_triples_response.py
      tests/python/test_ccsd_t_gradient_validation.py
      tests/python/test_ccsd_t_complete_gradient.py
      tests/python/test_dft_complete_cpu.py
      tests/python/test_ecp_stationary_cpu.py
      tests/python/test_ecp_f.py
      tests/python/test_ecp_f_projector.py
      tests/python/test_ecp_heavy.py
      tests/python/test_mp2_gradient.py
      tests/python/test_mp2_gradient_fast.py
      tests/python/test_hessian_analytic.py
      tests/python/test_hessian_directional.py
      tests/python/test_response_operator.py
      tests/python/test_hessian_reference.py
      tests/python/test_hessian_hvp.py
    )
    ;;
  ecp-forces)
    # Full d-force qualification exceeded the 20-minute job cap when
    # serialized in one file. Keep every case, split representations
    # across loadfile workers, and retain the existing job time cap.
    test_targets=(
      tests/python/test_ecp_public_cpu.py
      tests/python/test_ecp_spd_cartesian_cpu.py
      tests/python/test_ecp_spd_spherical_cpu.py
      tests/python/test_ecp_stuttgart_dft.py
      tests/python/test_ecp_halogen_dft.py
    )
    # This complete force matrix arrived on master after this router's
    # branch point. Route it whenever the checked-out Git tree owns it;
    # do not add a nonexistent target on the earlier branch itself.
    # Keep its 17 tests together under loadfile instead of late in core.
    # Preserve every functional, spin, lifecycle and finite difference.
    if git ls-files --error-unmatch -- tests/python/test_libxc_automatic_forces.py >/dev/null 2>&1; then
      test_targets+=(tests/python/test_libxc_automatic_forces.py)
    fi
    # This later master addition has 46 cases and ~17 minutes of serial
    # worker cost. Loadfile starts it on an initial worker; the 17-case
    # Libxc file follows the ~240 s public-ECP file. Measured file costs
    # estimate a ~20 minute force-shard tail, not a hosted timing claim.
    # Preserve all CPU/CUDA cases, oracles and cold/warm/rebound checks.
    if git ls-files --error-unmatch -- tests/python/test_mixed_rsh_df_forces.py >/dev/null 2>&1; then
      test_targets+=(tests/python/test_mixed_rsh_df_forces.py)
    fi
    ;;
  compiler-heavy)
    test_targets=(
      tests/python/test_codegen_high_l.py
      tests/python/test_codegen_production.py
      tests/python/test_codegen_cuda_compile.py
      tests/python/test_codegen_autotune.py
      tests/python/test_codegen_profiles.py
      tests/python/test_d3_geometry_ir.py
      tests/python/test_d3_generated_ragged.py
      tests/python/test_high_angular.py
      tests/python/test_one_electron_derivatives.py
      tests/python/test_range_separation.py
      tests/python/test_second_derivatives_inputs.py
      tests/python/test_derivative_aot_audit.py
      tests/python/test_eri_cpu_codegen.py
      tests/python/test_rccsdt_native_codegen_dependencies.py
      tests/python/test_df_cc_native_codegen.py
      tests/python/test_xc_retirement.py
      tests/python/test_stationary_cooperative_kernel_host.py
      tests/python/test_native_contraction_binding.py
      tests/python/test_rccsd_arena_liveness.py
      tests/python/test_cc_cuda_scalar_reduction_codegen.py
      tests/python/test_rccsd_response_admission.py
      tests/python/test_rccsd_iteration_reuse.py
      tests/python/test_stationary_cuda_lowering.py
      tests/python/test_rccsdt_native_triples_response_owner.py
      tests/python/test_becke_normalize_cooperative.py
      tests/python/test_rccsd_numeric_capacity.py
      tests/python/test_stationary_aot_generation.py
      tests/python/test_cc_shared_iteration_driver.py
      tests/python/test_rccsd_codegen_no_numpy.py
      tests/python/test_dft_mp_v1_capacity.py
      tests/python/test_df_lambda_frontier.py
    )
    if [[ "$GITHUB_EVENT_NAME" == "schedule" || "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]; then
      test_targets+=(tests/python/test_wb97mv_complete.py)
    fi
    ;;
  *)
    echo "::error::unknown Python shard: ${GENERATIVEQC_CI_SHARD}"
    exit 2
    ;;
esac

if [[ "${GITHUB_EVENT_NAME}" == "schedule" || "${GITHUB_EVENT_NAME}" == "workflow_dispatch" ]]; then
  case "${GENERATIVEQC_CI_SHARD}" in
    runtime-heavy)
      # Preserve one worker per complete Hessian module fixture.
      test_targets+=("${nightly_hessian_paths[@]}")
      ;;
    posthf)
      declare -A nightly_hessian_lookup=()
      for path in "${nightly_hessian_paths[@]}"; do
        nightly_hessian_lookup["${path}"]=1
      done
      kept_targets=()
      for path in "${test_targets[@]}"; do
        if [[ -z "${nightly_hessian_lookup[${path}]:-}" ]]; then
          kept_targets+=("${path}")
        fi
      done
      test_targets=("${kept_targets[@]}")
      ;;
  esac
fi

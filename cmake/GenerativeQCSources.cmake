include_guard(GLOBAL)

# Keep production source enumeration explicit and component-owned. Generated
# outputs are registered separately in GenerativeQCGeneratedSources.cmake.
function(generativeqc_add_api_sources target)
  set_property(SOURCE src/api/c_api_ks_hessian.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "GENERATIVEQC_HAS_NATIVE_HESSIAN=$<BOOL:${GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES}>")
  target_sources(${target} PRIVATE
    src/api/c_api_batch.cpp
    src/api/c_api_calculation.cpp
    src/api/c_api_integral_ad.cpp
    src/api/c_api_prepared_execution.cpp
    src/api/c_api_ks_snapshot.cpp
    src/api/c_api_ks_hessian.cpp
    src/api/c_api_fock.cpp
    src/api/c_api_one_electron_gradient.cpp
    src/api/c_api_projection.cpp
    src/api/c_api_hf_physical_audit.cpp
    src/api/c_api_response.cpp
    src/api/c_api_df_gradient.cpp
    src/api/c_api_d3.cpp
    src/api/c_api_d4.cpp
    src/api/c_api_nonlocal.cpp
    src/api/c_api_gcp.cpp
    src/api/c_api_common.cpp
    src/api/c_api_context.cpp
    src/api/c_api_basis_catalog.cpp
    src/api/c_api_basis_builtin.cpp
    src/api/c_api_resources.cpp
    src/api/c_api_resource_planning.cpp
    src/api/c_api_resource_ledger_plan.cpp
    src/api/c_api_device_resources.cpp
    src/api/c_api_scf_effort.cpp
    src/api/c_api_hf_verification.cpp
    src/api/c_api_hf_plan.cpp
    src/api/c_api_correlation_convergence.cpp
    src/api/c_api_response_recycle.cpp
    src/api/c_api_tuning.cpp
    src/api/c_api_ecp.cpp
    src/api/error.cpp
    src/methods/registry.cpp
    src/methods/named_ks_catalog.cpp
    src/methods/prepared_execution.cpp)
endfunction()

function(generativeqc_add_runtime_sources target)
  target_sources(${target} PRIVATE
    src/molecule/basis_catalog.cpp
    src/molecule/basis_template.cpp
    src/runtime/context.cpp
    src/runtime/ragged_topology.cpp
    src/runtime/resource_planning.cpp
    src/runtime/resource_ledger_plan.cpp
    src/runtime/cuda_provider.cpp
    src/tensor/cpu_linalg.cpp
    src/tensor/cpu/lp64_provider.cpp)
  if(GENERATIVEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/runtime/cuda_runtime.cu
      src/tensor/cuda_vector_contraction.cpp
      src/runtime/cuda_device_facts.cpp
      src/runtime/cuda_component_trace.cpp)
    if(CMAKE_DL_LIBS)
      target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
  endif()
endfunction()

function(generativeqc_add_dft_sources target)
  target_sources(${target} PRIVATE
    src/dft/ks_final_state.cpp
    src/methods/dft_method.cpp
    src/methods/dft_method_preparation.cpp
    src/methods/dft_method_final_state_adapter.cpp
    src/methods/dft_method_derivative_adapter.cpp)
  if(GENERATIVEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/dft/cosx_fock_provider.cpp
      src/dft/cosx_scf.cpp
      src/dft/cuda_cosx.cu
      src/dft/cuda_cosx_derivative.cu
      src/dft/cuda_xc.cpp
      src/dft/cuda_ks.cpp
      src/dft/cuda_stationary_controller.cpp
      src/dft/cuda_ks_kernels.cu
      src/dft/cuda_quadrature.cu
      src/dft/dispersion/d3_cuda.cu
      src/dft/dispersion/d4_cuda.cu
      src/dft/dispersion/gcp_cuda.cu
      src/dft/dispersion/d4_runtime_cuda.cu
      src/dft/nonlocal_correlation/vv10_runtime_cuda.cu)
    # Preserve Becke tail behavior; reassociation/FMA changes cancellation at
    # saturated partition boundaries. Independent grids qualify this policy.
    if(GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
      set_property(SOURCE src/dft/cuda_quadrature.cu APPEND PROPERTY
                   COMPILE_OPTIONS "-ffp-contract=off")
    else()
      set_property(SOURCE src/dft/cuda_quadrature.cu APPEND PROPERTY
                   COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
    endif()
  else()
    target_sources(${target} PRIVATE src/dft/cuda_stationary_binding_stub.cpp)
  endif()
endfunction()

function(generativeqc_add_posthf_cc_sources target)
  target_sources(${target} PRIVATE
    src/methods/mp2_method.cpp
    src/methods/correlated_batch.cpp
    src/methods/ump2_method.cpp
    src/methods/rccsd_method.cpp
    src/methods/uccsd_method.cpp
    src/methods/rccsdt_method.cpp
    src/methods/df_rccsdt_method.cpp
    src/methods/df_hf_guess.cpp
    src/hf/df_reference.cpp
    src/posthf/bridge.cpp
    src/posthf/cuda_derivative.cpp
    src/posthf/mp2_derivative_cuda.cpp
    src/methods/gfn2_runtime_bridge.cpp
    src/methods/xtb_method.cpp)
  if(GENERATIVEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/methods/df_ccsdt_force.cu
      src/hf/rhf_frame_response.cu
      src/cc/df_source_cuda.cu
      src/posthf/df_mo_response.cu
      src/cc/df_triples_cuda.cu
      src/cc/df_lambda_cuda.cu
      src/cc/cuda_solver.cu
      src/cc/lambda_response_cuda.cu
      src/cc/triples_response_cuda.cu
      src/cc/triples_fock_response_cuda.cu
      src/posthf/df_bridge.cu
      src/posthf/cuda_transform.cu
      src/posthf/ri_mp2_cuda.cu)
  endif()
endfunction()

function(generativeqc_add_integrals_scf_sources target)
  target_sources(${target} PRIVATE
    src/methods/hf_method.cpp
    src/scf/density_fitting.cpp
    src/scf/hf_df_resources.cpp
    src/scf/fleet.cpp
    src/scf/cuda_fock_provider.cpp
    src/scf/fock_prepared.cpp
    src/scf/explicit_integral_backend.cpp
    src/scf/rhf.cpp
    src/scf/cuda/df_source_domain.cpp
    src/scf/cuda/rhf_policy.cpp)

  if(GENERATIVEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/integrals/ecp_cuda.cu
      src/scf/cuda/df_derivatives.cu
      src/scf/cuda/df_shell_derivatives.cu
      src/scf/cuda/df_gradient_bridge.cu
      src/scf/cuda/df_response_weights.cu
      src/scf/cuda/one_electron_values.cu
      src/scf/cuda/one_electron_derivatives.cu
      src/scf/cuda/one_electron_gradient_bridge.cu
      src/scf/cuda/df_coulomb.cpp
      src/scf/cuda/df_exchange.cpp
      src/scf/cuda/df_occupied_exchange.cpp
      src/scf/cuda/df_scf_factor.cpp
      src/scf/cuda/df_scf_final_state.cpp
      src/scf/cuda/df_scf_warm.cpp
      src/scf/cuda/df_force_response.cpp
      src/scf/cuda/df_jk.cpp
      src/scf/cuda/df_jk_kernels.cu
      src/scf/cuda/df_packed_values.cu
      src/scf/cuda/df_metric_kernels.cu
      src/scf/cuda/df_plan.cpp
      src/scf/cuda/df_plan_lifetime.cpp
      src/scf/cuda/df_plan_setup.cpp
      src/scf/cuda/df_rhf_scf.cpp
      src/scf/cuda/df_runtime.cpp
      src/scf/cuda/df_scf_kernels.cu
      src/scf/cuda/df_scf_library.cpp
      src/scf/cuda/df_scf_diis.cpp
      src/scf/cuda/df_eigensystem.cpp
      src/scf/cuda/df_final_validation.cpp
      src/scf/cuda/resident_final_validation.cpp
      src/scf/cuda/final_validation_kernels.cu
      src/scf/cuda/df_uhf_scf.cpp
      src/scf/cuda_eigensolver_probe.cu
      src/scf/cuda_rhf.cpp
      src/scf/cuda/rhf_bucket.cpp
      src/scf/cuda/hf_initial_guess.cpp
      src/scf/cuda/warm_density.cpp
      src/scf/cuda/rhf_source_handoff.cpp
      src/scf/cuda/rhf_graph.cpp
      src/scf/cuda/rhf_resident_values.cpp
      src/scf/cuda/direct_jk_source_lease.cpp
      src/scf/cuda/one_electron_reference.cu
      src/scf/cuda/nuclear_kernels.cu
      src/scf/cuda/direct_pair_cache.cu
      src/scf/cuda/direct_jk.cpp
      src/scf/cuda/direct_coulomb.cpp
      src/scf/cuda_fock_execution.cpp
      src/scf/cuda/one_electron_view.cpp
      src/scf/cuda/one_electron_export.cpp
      src/scf/cuda/one_electron_export_batch.cpp
      src/scf/cuda/direct_tile_validation.cu
      src/scf/cuda/direct_density_bounds.cu
      src/scf/cuda/direct_tile_compaction.cu
      src/scf/cuda/direct_generated_tasks.cu
      src/scf/cuda/direct_resident_tasks.cu
      src/scf/cuda/direct_bounded_pages.cu
      src/scf/cuda/direct_bounded_tasks.cu
      src/scf/cuda/direct_queue_scan.cu
      src/scf/cuda/direct_queue_diagnostics.cu
      src/scf/cuda/resources.cpp
      src/scf/cuda/matrix_library.cpp
      src/tensor/cuda_matrix_product.cpp
      src/scf/cuda/runtime_support.cpp
      src/scf/cuda/scf_state_kernels.cu
      src/scf/cuda/scf_matrix_kernels.cu
      src/scf/cuda/mean_field_setup.cu
      src/scf/cuda/scf_density_kernels.cu
      src/scf/cuda/structured_matrix.cu
      src/scf/cuda/scf_diis_kernels.cu
      src/scf/cuda/scf_convergence_kernels.cu
      src/scf/cuda/basis_transform_kernels.cu
      src/scf/cuda/df_source_setup.cpp
      src/scf/cuda/df_md_owner.cu
      src/scf/cuda/df_md_raw.cu
      src/scf/cuda/df_md_plan.cpp
      src/scf/cuda/df_source.cpp
      src/scf/cuda/df_source_kernels.cu
      src/scf/cuda/df_integral_export.cpp
      src/scf/cuda/df_integral_export_batch.cpp
      src/scf/cuda/arena.cpp
      src/scf/cuda/topology.cpp
      src/scf/cuda/queue_plan.cpp
      src/scf/cuda/queue_profile.cpp
      src/scf/cuda/eigensolver.cpp
      src/solver/cuda/generalized_eigen.cpp
      src/solver/cuda/symmetric_eigen_provider.cpp
      src/solver/cuda/cholesky_provider.cpp
      src/solver/cuda/symmetric_eigen_handles.cpp
      src/solver/cuda/symmetric_eigen_workspace.cpp
      src/scf/cuda/eigensolver_kernels.cu
      src/scf/cuda/explicit_integral_backend.cu
      src/scf/cuda_hf_entry.cpp)
  else()
    target_sources(${target} PRIVATE
      src/scf/cuda_density_fitting_stub.cpp
      src/integrals/ecp_cuda_stub.cpp
      src/scf/cuda_direct_jk_stub.cpp
      src/scf/cuda_density_fitting_integrals_stub.cpp
      src/scf/cuda_weighted_eri_stub.cpp)
  endif()
endfunction()

function(generativeqc_add_component_sources target)
  generativeqc_add_api_sources(${target})
  generativeqc_add_runtime_sources(${target})
  generativeqc_add_integrals_scf_sources(${target})
  generativeqc_add_dft_sources(${target})
  generativeqc_add_posthf_cc_sources(${target})
endfunction()

# Internal CPU-only compile boundaries. Keep CUDA, generated-header consumers,
# and device-link owners on the parent target until their dependencies are split.
# OBJECT libraries preserve every compiled object in both the shared SDK/wheel
# and the static monolithic CLI; ordinary STATIC archives could omit objects.
# OBJECT boundaries compile host C++ once and inject all symbols into the
# published library. Explicit generated-header edges keep parallel Ninja builds
# correct without depending on every code generator through the parent.
function(generativeqc_add_host_object_component parent component)
  cmake_parse_arguments(PARSE_ARGV 2 HOC "" "" "SOURCES;CODEGEN")
  if(HOC_UNPARSED_ARGUMENTS OR HOC_KEYWORDS_MISSING_VALUES OR NOT HOC_SOURCES)
    message(FATAL_ERROR "${component}: expected SOURCES and optional CODEGEN")
  endif()
  if(NOT TARGET ${parent})
    message(FATAL_ERROR "${component}: missing parent library ${parent}")
  endif()
  get_target_property(_parent_sources ${parent} SOURCES)
  get_property(_claimed GLOBAL PROPERTY GENERATIVEQC_HOST_COMPONENT_SOURCES)
  foreach(_source IN LISTS HOC_SOURCES)
    if(NOT _source MATCHES "\\.cpp$")
      message(FATAL_ERROR "${component}: only host .cpp units belong in OBJECT components: ${_source}")
    endif()
    if(_source IN_LIST _parent_sources OR _source IN_LIST _claimed)
      message(FATAL_ERROR "${component}: duplicate production translation unit ${_source}")
    endif()
    list(APPEND _claimed "${_source}")
  endforeach()
  set_property(GLOBAL PROPERTY GENERATIVEQC_HOST_COMPONENT_SOURCES "${_claimed}")

  add_library(${component} OBJECT ${HOC_SOURCES})
  set_target_properties(${component} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${component} PRIVATE
    "$<TARGET_PROPERTY:${parent},INCLUDE_DIRECTORIES>")
  target_compile_definitions(${component} PRIVATE
    "$<TARGET_PROPERTY:${parent},COMPILE_DEFINITIONS>")
  target_compile_options(${component} PRIVATE
    "$<TARGET_PROPERTY:${parent},COMPILE_OPTIONS>")
  target_link_libraries(${component} PRIVATE Threads::Threads)

  # The shared/wheel CUDA parent can obtain toolkit includes through its
  # link interfaces; OBJECT compilation has no such implicit link inheritance.
  if(GENERATIVEQC_ENABLE_CUDA)
    target_include_directories(${component} PRIVATE
      ${GENERATIVEQC_CUDA_TOOLKIT_INCLUDE_DIRS})
  endif()
  foreach(_generator IN LISTS HOC_CODEGEN)
    if(NOT TARGET ${_generator})
      message(FATAL_ERROR "${component}: missing generated-header owner ${_generator}")
    endif()
    add_dependencies(${component} ${_generator})
  endforeach()
  if(GENERATIVEQC_ENABLE_CXX_PCH)
    target_precompile_headers(${component} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/pch.hpp>")
  endif()
  target_sources(${parent} PRIVATE $<TARGET_OBJECTS:${component}>)
endfunction()

function(generativeqc_add_cpu_component_object_targets target)
  generativeqc_add_host_object_component(${target} generativeqc_cpu_scf_solver
    SOURCES
      src/scf/reference/linalg.cpp
      src/scf/solver/proposal_control.cpp
      src/scf/solver/eigen_frame.cpp
      src/scf/solver/cpu_target_eigen.cpp
      src/scf/solver/final_state.cpp
      src/scf/solver/warm_subspace.cpp
      src/solver/cpu/prepared_spectral.cpp)
  generativeqc_add_host_object_component(${target} generativeqc_cpu_basis_integrals
    SOURCES
      src/molecule/basis.cpp
      src/integrals/minao_basis.cpp
      src/integrals/density_fitting_metric.cpp)
  generativeqc_add_host_object_component(${target} generativeqc_host_integrals
    SOURCES
      src/integrals/s_integrals.cpp
      src/integrals/generated_df_cpu.cpp
      src/integrals/ecp.cpp
      src/integrals/stationary_component_contract.cpp
    CODEGEN
      generativeqc_eri_cpu_codegen
      generativeqc_one_electron_st_cpu_codegen
      generativeqc_stationary_cpu_dispatch_codegen
      generativeqc_df_values_cpu_codegen
      generativeqc_df_derivatives_cpu_codegen
      generativeqc_ecp_codegen)
  generativeqc_add_host_object_component(${target} generativeqc_host_scf
    SOURCES
      src/scf/density_factor.cpp
      src/scf/df_response_weights.cpp
      src/scf/fock_build.cpp
      src/scf/fock_provider.cpp
      src/scf/fock_execution.cpp
      src/scf/stationary_integral_gradient.cpp
      src/scf/solver/diis.cpp
      src/scf/solver/mean_field_driver.cpp
      src/solver/cpu/johnson_broyden.cpp
      src/scf/gradient/hf_gradient.cpp
      src/scf/reference/mean_field.cpp
      src/scf/initial_guess/density.cpp
      src/scf/occupied_projection.cpp
      src/scf/initial_guess/minao.cpp
      src/scf/initial_guess/huckel.cpp
      src/scf/preliminary_guess.cpp
      src/tensor/symmetric_matrix_function.cpp
    CODEGEN
      generativeqc_scf_array_cpu_codegen
      generativeqc_broyden_cpu_codegen
      generativeqc_stationary_integral_weights_codegen)
  generativeqc_add_host_object_component(${target} generativeqc_host_dft
    SOURCES
      src/dft/ao_grid.cpp
      src/dft/cosx_reference.cpp
      src/dft/bridge.cpp
      src/dft/grid.cpp
      src/dft/rks.cpp
      src/dft/uks.cpp
      src/dft/xc.cpp
      src/dft/dispersion/d3_runtime.cpp
      src/dft/dispersion/d4_table_data.cpp
      src/dft/dispersion/d4_runtime.cpp
      src/dft/nonlocal_correlation/vv10_runtime.cpp
      src/dft/nonlocal_correlation/vv10_integration.cpp
    CODEGEN
      generativeqc_xc_cpu_codegen
      generativeqc_d3_codegen
      generativeqc_d4_derivative_codegen
      # d4_reference.hpp includes the shared method-parameter constants.
      generativeqc_method_parameters_codegen
      generativeqc_nonlocal_pair_codegen
      generativeqc_libxc_semilocal_cpu_codegen)
  generativeqc_add_host_object_component(${target} generativeqc_host_posthf_cc
    SOURCES
      src/cc/solver.cpp
      src/cc/lambda_response.cpp
      src/cc/triples_response.cpp
      src/cc/triples_fock_response.cpp
      src/cc/rccsdt_force.cpp
      src/cc/uccsd_solver.cpp
      src/posthf/mp2_derivative_common.cpp
      src/posthf/mp2_derivative_cpu.cpp
      src/posthf/mp2_energy.cpp
      src/posthf/ump2_energy.cpp
      src/posthf/ump2_force.cpp
      src/posthf/mp2_force.cpp
      src/posthf/mp2_gradient.cpp
      src/posthf/native_provider.cpp
      src/response/native_gmres.cpp
      src/response/resident_gmres.cpp
    CODEGEN
      generativeqc_rccsd_cpu_codegen
      generativeqc_uccsd_cpu_codegen
      # df_plan.hpp -> hoisted -> core -> native generated headers.
      generativeqc_df_ccsd_native_codegen
      generativeqc_df_ccsd_core_codegen
      generativeqc_df_ccsd_hoisted_codegen
      generativeqc_triples_fock_cpu_codegen
      generativeqc_df_mo_source_codegen)
endfunction()

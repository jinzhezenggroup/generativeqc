include_guard(GLOBAL)

function(generativeqc_add_gfn2_runtime target)
  set(_gfn2_root "${CMAKE_CURRENT_SOURCE_DIR}/src/xtb/native")
  target_sources(${target} PRIVATE
    ${_gfn2_root}/src/model/common/integrals_kernels_baseline.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels_baseline.cpp
    ${_gfn2_root}/src/model/common/sto.cpp
    ${_gfn2_root}/src/model/common/integrals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/methods/gfn2_electronic_update.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_embedding.cpp
    ${_gfn2_root}/src/model/gfn2/es2.cpp
    ${_gfn2_root}/src/backends/common/gfn2_plan_schema.cpp
    ${_gfn2_root}/src/model/gfn2/aes2.cpp
    ${_gfn2_root}/src/model/gfn2/basis.cpp
    ${_gfn2_root}/src/model/gfn2/coordination.cpp
    ${_gfn2_root}/src/model/gfn2/d4.cpp
    ${_gfn2_root}/src/model/gfn2/es3.cpp
    ${_gfn2_root}/src/model/gfn2/external_point_charges.cpp
    ${_gfn2_root}/src/model/gfn2/force.cpp
    ${_gfn2_root}/src/model/gfn2/h0.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels.cpp
    ${_gfn2_root}/src/model/gfn2/repulsion.cpp
    ${_gfn2_root}/src/model/gfn2/scc_driver.cpp
    ${_gfn2_root}/src/model/gfn2/scc_mixer.cpp
    ${_gfn2_root}/src/model/gfn2/spin.cpp
    ${_gfn2_root}/src/model/gfn2/wavefunction.cpp
    ${_gfn2_root}/src/runtime/gfn2_cpu_execution.cpp)

  # Molecular CUDA execution consumes compiler-owned science and native
  # allocation/SCC/solver owners. The provenance manifest retains the original
  # CPU/CUDA snapshot identities and records the adapted source subset.
  if(GENERATIVEQC_ENABLE_CUDA AND NOT GENERATIVEQC_PYTHON_WHEEL AND
     NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
    set(_gfn2_cuda_sources
      ${_gfn2_root}/src/runtime/cuda_descriptor_validation.cu
      ${_gfn2_root}/src/runtime/gfn2_cuda_topology_staging.cu
      ${_gfn2_root}/src/runtime/gfn2_cuda_execution.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_aes2.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_classical_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_d4.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_density.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_electric_field.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_electronic_gradient.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_energy_force_execution.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_eigensolver.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_es2.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_es3.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_external_point_charges.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_force_composition.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_geometry.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_preprocessing.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_public_result_bridge.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_h0_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_hamiltonian.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_hamiltonian_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_inference_publication.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_integrals.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_mulliken.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_occupations.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_pairlist.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_parameters.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_plan_schema.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_periodic_embedding.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_post_scc_potential.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_repulsion.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_bridge.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_classical_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_free_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_loop.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_arena.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_control.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_initialize.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_reports.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_mixer.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_potential.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_publication.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_eigensolver.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_inputs.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_topology.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_spin.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_terminal_classical_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_total_energy.cu
    )
    add_library(generativeqc_gfn2_cuda STATIC ${_gfn2_cuda_sources})
    option(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS
      "Compile bounded GFN2 device density work receipts" OFF)
    if(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)
      target_compile_definitions(generativeqc_gfn2_cuda PRIVATE
        GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1)
      # The parent target compiles gfn2_runtime_bridge.cpp, whose execution
      # header includes the layout-bearing density workspace. Keep that TU's
      # opt-in artifact ABI identical to the whole-archived CUDA target.
      target_compile_definitions(${target} PRIVATE
        GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1)
    endif()
    set(GENERATIVEQC_GFN2_ELECTRONIC_CUDA_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_electronic_native.cuh")
    generativeqc_register_generated_sources(
      NAME generativeqc_gfn2_electronic_cuda_codegen
      TARGET generativeqc_gfn2_cuda
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_electronic_cuda.py"
      OUTPUTS "${GENERATIVEQC_GFN2_ELECTRONIC_CUDA_HEADER}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_electronic.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_electronic_contract.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_electronic_runtime.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_electronic_schedule.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/provenance.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scf.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/weighted_gram.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/weighted_gram_emit.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/native_lowering.py"
      ARGS --output "${GENERATIVEQC_GFN2_ELECTRONIC_CUDA_HEADER}"
      COMMENT "Generating compiler-owned GFN2 CUDA electronic pair science")
    generativeqc_register_generated_sources(
      NAME generativeqc_gfn2_density_cuda_codegen
      TARGET generativeqc_gfn2_cuda
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_density_cuda.py"
      OUTPUTS
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_density_contract.inc"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_density_contract_receipt.inc"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_density_lowering.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/weighted_gram.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/weighted_gram_emit.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/native_lowering.py"
      ARGS --output "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_density_contract.inc"
           --instrumented-output
           "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_density_contract_receipt.inc"
      COMMENT "Generating shared paired GFN2 weighted-Gram contraction")
    # Both CPU and CUDA consume compiler-owned pair and D4 parameter artifacts.
    add_dependencies(generativeqc_gfn2_cuda
      generativeqc_method_parameters_codegen
      generativeqc_gfn2_pair_cpu_codegen
      generativeqc_gfn2_sdq_cuda_codegen
      generativeqc_gfn2_es2_native_codegen
      generativeqc_gfn2_external_point_charge_force_native_codegen)
    set(_gfn2_aes2_cuda_header
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_aes2_native.cuh")
    generativeqc_register_generated_sources(
      NAME generativeqc_gfn2_aes2_cuda_codegen
      TARGET generativeqc_gfn2_cuda
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_aes2_native.py"
      OUTPUTS "${_gfn2_aes2_cuda_header}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_aes2.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_aes2_schedule.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
      ARGS --cuda-output "${_gfn2_aes2_cuda_header}"
      COMMENT "Generating compiler-owned GFN2 AES2 CUDA kernels")
    target_include_directories(generativeqc_gfn2_cuda PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
      ${_gfn2_root}
      ${_gfn2_root}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(generativeqc_gfn2_cuda PRIVATE GENERATIVEQC_XTB_HAS_CUDA=1)
    set_target_properties(generativeqc_gfn2_cuda PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if(NOT "${_generativeqc_cuda_compile_pool}" STREQUAL "")
      set_property(TARGET generativeqc_gfn2_cuda PROPERTY JOB_POOL_COMPILE
                   "${_generativeqc_cuda_compile_pool}")
    endif()
    set_source_files_properties(
      ${_gfn2_root}/src/backends/cuda/gfn2_pairlist.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_geometry.cu
      PROPERTIES COMPILE_OPTIONS "-fmad=false")
    target_compile_definitions(${target} PRIVATE GENERATIVEQC_HAS_GFN2_CUDA=1)

    # The CUDA archive resolves its own device symbols. Consume the complete
    # archive so CUDA registration/device-link objects cannot be discarded,
    # without propagating separable compilation to unrelated GenerativeQC CUDA TUs.
    target_link_libraries(${target} PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,generativeqc_gfn2_cuda>")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      generativeqc_attach_cuda_driver_implib(${target})
    else()
      target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
    endif()
  endif()

  target_include_directories(${target} PRIVATE
    ${_gfn2_root}
    ${_gfn2_root}/src)
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()


endfunction()

function(generativeqc_link_cli_core target)
  if(GENERATIVEQC_MONOLITHIC_CLI AND TARGET generativeqc_gfn2_cuda AND
     (CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED OR CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED))
    # The CUDA archive calls host GFN2 helpers and lazy driver imports in the
    # static core. Rescan this pair at the final link, retaining every CUDA
    # registration/device-link object without whole-archiving the entire core.
    target_link_libraries(${target} PRIVATE
      "$<LINK_GROUP:RESCAN,generativeqc,$<LINK_LIBRARY:WHOLE_ARCHIVE,generativeqc_gfn2_cuda>>")
  else()
    target_link_libraries(${target} PRIVATE generativeqc)
  endif()
endfunction()

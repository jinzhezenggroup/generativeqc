include_guard(GLOBAL)

# This target intentionally has no CUDA or public runtime dependency. Consumers
# must still supply a qualified stationary/response owner before claiming a
# complete molecular Hessian; linking these scalar leaves alone is insufficient.
function(generativeqc_add_host_hessian_aot target inventory)
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Host Hessian AOT generation requires build-time Python")
  endif()
  get_filename_component(_inventory "${inventory}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${_inventory}" OR IS_DIRECTORY "${_inventory}")
    message(FATAL_ERROR "Explicit host Hessian AOT inventory is missing: ${_inventory}")
  endif()
  set(_generator "${PROJECT_SOURCE_DIR}/tools/generate_host_hessian_aot.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_inventory}" "${_generator}"
    "${PROJECT_SOURCE_DIR}/python/generativeqc_compiler/method/host_hessian_aot.py")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_generator}" --inventory "${_inventory}" --list
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _units ERROR_VARIABLE _error
    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 120)
  if(NOT _result EQUAL 0 OR "${_units}" STREQUAL "")
    message(FATAL_ERROR "Invalid host Hessian AOT inventory: ${_error}")
  endif()
  set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/host_hessian_aot")
  set(_sources)
  foreach(_unit IN LISTS _units)
    set(_output "${_directory}/${_unit}")
    generativeqc_register_generated_sources(
      OUTPUTS "${_output}" GENERATOR "${_generator}"
      DEPENDS "${_inventory}" "${PROJECT_SOURCE_DIR}/src/dft/host_hessian_aot.hpp"
      ARGS --inventory "${_inventory}" --unit "${_unit}" --output "${_output}"
      COMMENT "Generating CPU Hessian scalar AD leaf ${_unit}")
    list(APPEND _sources "${_output}")
  endforeach()
  set(_d3_tables "${_directory}/d3_data.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_d3_tables}"
    GENERATOR "${PROJECT_SOURCE_DIR}/tools/generativeqc_d3/generate_native_data.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/data/parameters/d3_production.bin"
      "${PROJECT_SOURCE_DIR}/manifests/xtbloom-d3.json"
    ARGS --output "${_d3_tables}"
    COMMENT "Generating pinned D3 tables for independently linked CPU columns")
  add_library(${target} STATIC ${_sources} "${_d3_tables}"
    "${PROJECT_SOURCE_DIR}/src/dft/host_xc_response.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/dispersion/host_d3_column.cpp")
  set_target_properties(${target} PROPERTIES GENERATIVEQC_HOST_SCALAR_INVENTORY "${_inventory}")
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(${target} SYSTEM PRIVATE "${_directory}")
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-fast-math;-ffp-contract=off>
    $<$<CXX_COMPILER_ID:MSVC>:/fp:strict>)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# The coupled consumer borrows the ordinary native mean-field implementation;
# it is separate from the independently linkable scalar/point package.
function(generativeqc_add_host_ks_response target leaves core)
  cmake_parse_arguments(_host "LINK_INTO_CORE" "" "" ${ARGN})
  if(NOT TARGET "${leaves}" OR NOT TARGET "${core}")
    message(FATAL_ERROR "Host KS response requires explicit linked leaves and native core targets")
  endif()
  if(GENERATIVEQC_ENABLE_CUDA AND NOT _host_LINK_INTO_CORE)
    message(FATAL_ERROR "Independently linked host KS response requires a CPU-only native core")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Host KS response generation requires build-time Python")
  endif()
  if(DEFINED GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT AND
      NOT GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT)
    message(FATAL_ERROR "Host nuclear integral response requires linked stationary CPU primitive AD")
  endif()
  get_target_property(_inventory ${leaves} GENERATIVEQC_HOST_SCALAR_INVENTORY)
  if(NOT _inventory)
    message(FATAL_ERROR "Host nuclear integral response requires the exact leaf inventory")
  endif()
  set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/host_ks_response")
  set(_maps "${_directory}/host_ks_cpks_maps.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_maps}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_host_ks_cpks.py"
    ARGS --output "${_maps}"
    COMMENT "Generating CPU KS metric, RHS and density/weight response maps")
  set(_grid_maps "${_directory}/host_grid_maps.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_grid_maps}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_host_grid_maps.py"
    ARGS --output "${_grid_maps}"
    COMMENT "Generating CPU moving-grid AO feature and potential maps")
  set(_integral_maps "${_directory}/host_nuclear_integral_maps.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_integral_maps}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_host_nuclear_integrals.py"
    DEPENDS "${_inventory}"
    ARGS --inventory "${_inventory}" --output "${_integral_maps}"
    COMMENT "Generating method-bound CPU nuclear integral contractions")
  set(_curvature_maps "${_directory}/host_xc_curvature_maps.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_curvature_maps}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_host_grid_maps.py"
    ARGS --curvature --output "${_curvature_maps}"
    COMMENT "Generating fixed-density CPU GGA mixed AO feature maps")
  set(_geometry_ad "${_directory}/host_grid_geometry_ad.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_geometry_ad}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_grid_geometry_response.py"
    ARGS --backend cpu --output "${_geometry_ad}"
    COMMENT "Generating shared CPU mixed AO and partition geometry AD")
  set(_one_electron "${_directory}/generated_one_electron_cpu.hpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_one_electron}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    ARGS --cpu-st-output "${_one_electron}"
    COMMENT "Generating shared CPU s/p/d/f S/T/V primitive AD")
  set(_one_electron_mixed "${_directory}/one_electron_mixed.cpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_one_electron_mixed}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_one_electron_mixed.py"
    ARGS --output "${_one_electron_mixed}"
    COMMENT "Generating bounded CPU s/p/d/f mixed S/T/V primitive AD")
  set(_factorized_second "${_directory}/host_factorized_second.hpp")
  set(_factorized_first "${_directory}/host_factorized_first.cpp")
  set(_factorized_value "${_directory}/host_factorized_value.cpp")
  set(_factorized_matrix "${_directory}/host_factorized_matrix.cpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_factorized_second}" "${_factorized_first}" "${_factorized_value}" "${_factorized_matrix}"
    GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_factorized_second.py"
    ARGS --output "${_factorized_second}" --first-output "${_factorized_first}" --value-output "${_factorized_value}" --matrix-output "${_factorized_matrix}"
    COMMENT "Generating canonical CPU full-range ERI value, first, mixed and matrix primitives")
  add_library(${target} STATIC "${PROJECT_SOURCE_DIR}/src/dft/host_ks_response.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_ks_perturbation.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_grid_geometry.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_grid_mixed_geometry.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_grid_column_geometry.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_xc_nuclear.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_xc_curvature.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_xc_column.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_nuclear_integrals.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_ks_nuclear_response.cpp"
    "${PROJECT_SOURCE_DIR}/src/dft/host_ks_nuclear_hessian.cpp"
    "${_maps}" "${_grid_maps}" "${_integral_maps}" "${_geometry_ad}" "${_curvature_maps}" "${_one_electron}" "${_one_electron_mixed}" "${_factorized_second}" "${_factorized_first}" "${_factorized_value}" "${_factorized_matrix}")
  target_include_directories(${target} SYSTEM PRIVATE "${_directory}")
  target_link_libraries(${target} PUBLIC "${leaves}")
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/include")
  if(_host_LINK_INTO_CORE)
    target_link_libraries(${core} PRIVATE ${target})
    target_compile_definitions(${target} PRIVATE
      GENERATIVEQC_HAS_CUDA=$<BOOL:${GENERATIVEQC_ENABLE_CUDA}>)
    if(GENERATIVEQC_ENABLE_CUDA)
      target_link_libraries(${target} PRIVATE CUDA::cudart)
    endif()
  else()
    target_link_libraries(${target} PUBLIC "${core}")
  endif()
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-fast-math;-ffp-contract=off>
    $<$<CXX_COMPILER_ID:MSVC>:/fp:strict>)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

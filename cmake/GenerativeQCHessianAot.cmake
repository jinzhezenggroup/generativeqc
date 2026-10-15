include_guard(GLOBAL)

# Build an explicit leaf package only. This target does not register a native
# method/Hessian capability until a complete stationary owner consumes it.
function(generativeqc_add_hessian_aot target inventory)
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Hessian AOT generation requires build-time Python")
  endif()
  set(_generator "${PROJECT_SOURCE_DIR}/tools/generate_hessian_aot.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${inventory}" "${_generator}"
    "${PROJECT_SOURCE_DIR}/tools/hessian_aot_inventory.py")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_generator}"
            --inventory "${inventory}" --list-programs
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _programs ERROR_VARIABLE _error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0 OR "${_programs}" STREQUAL "")
    message(FATAL_ERROR "Invalid explicit Hessian AOT inventory: ${_error}")
  endif()
  set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/hessian_aot")
  set(_sources)
  foreach(_program IN LISTS _programs)
    set(_output "${_directory}/${_program}.cu")
    generativeqc_register_generated_sources(
      OUTPUTS "${_output}"
      GENERATOR "${_generator}"
      DEPENDS "${inventory}" "${PROJECT_SOURCE_DIR}/src/dft/hessian_aot.hpp"
      ARGS --inventory "${inventory}" --program "${_program}" --output "${_output}"
      COMMENT "Generating native Hessian AD leaf ${_program}")
    list(APPEND _sources "${_output}")
  endforeach()
  set(_registry "${_directory}/registry.cpp")
  generativeqc_register_generated_sources(
    OUTPUTS "${_registry}" GENERATOR "${_generator}" DEPENDS "${inventory}"
    ARGS --inventory "${inventory}" --registry --output "${_registry}"
    COMMENT "Generating explicit native Hessian AD leaf registry")
  set(_gmres "${CMAKE_CURRENT_BINARY_DIR}/generated/response/cuda_gmres_kernels.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_gmres}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_cuda_gmres.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/response/cuda_gmres.hpp"
    ARGS --output "${_gmres}"
    COMMENT "Generating numerically device-only native GMRES kernels")
  set(_cpks "${CMAKE_CURRENT_BINARY_DIR}/generated/response/ks_cpks_maps.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_cpks}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_ks_cpks.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/response/cuda_cpks.hpp"
    ARGS --output "${_cpks}"
    COMMENT "Generating compiler-AD native RKS/UKS response maps")
  set(_grid "${CMAKE_CURRENT_BINARY_DIR}/generated/response/wb97mv_grid_response.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_grid}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_wb97mv_grid_response.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/dft/cuda_wb97mv_grid.hpp"
    ARGS --output "${_grid}"
    COMMENT "Generating compiler-AD native WB97M-V grid response")
  set(_fock "${CMAKE_CURRENT_BINARY_DIR}/generated/response/wb97mv_fock_response.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_fock}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_wb97mv_fock_response.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/dft/cuda_ks_wb97mv_response.hpp"
    ARGS --output "${_fock}"
    COMMENT "Generating compiler-AD native WB97M-V physical Fock response")
  set(_geometry "${CMAKE_CURRENT_BINARY_DIR}/generated/response/grid_geometry_response.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_geometry}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_grid_geometry_response.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/dft/cuda_grid_geometry.hpp"
    ARGS --output "${_geometry}"
    COMMENT "Generating compiler-AD native moving-grid geometry response")
  set(_nuclear "${CMAKE_CURRENT_BINARY_DIR}/generated/response/wb97mv_nuclear_grid.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_nuclear}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_wb97mv_nuclear_grid.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/dft/cuda_wb97mv_grid.hpp"
    ARGS --output "${_nuclear}"
    COMMENT "Generating compiler-AD native WB97M-V moving-grid response")
  set(_one_electron "${CMAKE_CURRENT_BINARY_DIR}/generated/response/nuclear_integral_response.cuh")
  generativeqc_register_generated_sources(
    OUTPUTS "${_one_electron}" GENERATOR "${PROJECT_SOURCE_DIR}/tools/generate_nuclear_integral_response.py"
    DEPENDS "${PROJECT_SOURCE_DIR}/src/integrals/cuda_one_electron_response.hpp"
    ARGS --output "${_one_electron}"
    COMMENT "Generating compiler-AD native weighted integral response")
  add_library(${target} STATIC ${_sources} "${_registry}" "${_gmres}" "${_cpks}" "${_grid}" "${_fock}" "${_geometry}" "${_nuclear}" "${_one_electron}"
    "${PROJECT_SOURCE_DIR}/src/dft/hessian_aot.cpp"
    "${PROJECT_SOURCE_DIR}/src/response/cuda_cpks.cu"
    "${PROJECT_SOURCE_DIR}/src/dft/cuda_wb97mv_grid.cu"
    "${PROJECT_SOURCE_DIR}/src/dft/cuda_grid_geometry.cu"
    "${PROJECT_SOURCE_DIR}/src/dft/cuda_ks_wb97mv_response.cu"
    "${PROJECT_SOURCE_DIR}/src/dft/cuda_ks_one_electron_response.cu"
    "${PROJECT_SOURCE_DIR}/src/integrals/cuda_one_electron_response.cu"
    "${PROJECT_SOURCE_DIR}/src/response/cuda_gmres.cu")
  generativeqc_set_cuda_compile_pool(${target} "${_generativeqc_aot_compile_pool}")
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src"
    PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_link_libraries(${target} PUBLIC CUDA::cudart)
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  if(GENERATIVEQC_BUILD_TESTS)
    add_executable(generativeqc_hessian_aot_probe
      "${PROJECT_SOURCE_DIR}/tests/native/hessian_aot_probe.cpp")
    target_link_libraries(generativeqc_hessian_aot_probe PRIVATE ${target})
    add_executable(generativeqc_cuda_gmres_probe
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_gmres_probe.cu")
    target_link_libraries(generativeqc_cuda_gmres_probe PRIVATE ${target})
    target_compile_options(generativeqc_cuda_gmres_probe PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
    set_target_properties(generativeqc_cuda_gmres_probe PROPERTIES
      CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(generativeqc_cuda_cpks_probe
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_cpks_probe.cu")
    target_link_libraries(generativeqc_cuda_cpks_probe PRIVATE ${target})
    target_compile_options(generativeqc_cuda_cpks_probe PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
    set_target_properties(generativeqc_cuda_cpks_probe PROPERTIES
      CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(generativeqc_cuda_wb97mv_grid_probe
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_wb97mv_grid_probe.cpp")
    target_link_libraries(generativeqc_cuda_wb97mv_grid_probe PRIVATE ${target})
    add_executable(generativeqc_ks_wb97mv_grid_probe
      "${PROJECT_SOURCE_DIR}/tests/native/ks_wb97mv_grid_probe.cpp")
    target_link_libraries(generativeqc_ks_wb97mv_grid_probe PRIVATE ${target} generativeqc)
    add_executable(generativeqc_cuda_grid_geometry_probe
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_grid_geometry_probe.cpp")
    target_link_libraries(generativeqc_cuda_grid_geometry_probe PRIVATE ${target})
    add_executable(generativeqc_cuda_one_electron_probe
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_one_electron_probe.cpp")
    target_link_libraries(generativeqc_cuda_one_electron_probe PRIVATE ${target} generativeqc)
    add_executable(generativeqc_cuda_one_electron_closure_tests
      "${PROJECT_SOURCE_DIR}/tests/native/cuda_one_electron_closure_tests.cpp")
    target_link_libraries(generativeqc_cuda_one_electron_closure_tests PRIVATE ${target})
    add_test(NAME cuda_one_electron_host_closure COMMAND generativeqc_cuda_one_electron_closure_tests)
  endif()
endfunction()

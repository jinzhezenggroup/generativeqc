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
  add_library(${target} STATIC ${_sources} "${_registry}" "${_gmres}" "${_cpks}"
    "${PROJECT_SOURCE_DIR}/src/dft/hessian_aot.cpp"
    "${PROJECT_SOURCE_DIR}/src/response/cuda_cpks.cu"
    "${PROJECT_SOURCE_DIR}/src/response/cuda_gmres.cu")
  generativeqc_set_cuda_compile_pool(${target} "${_generativeqc_aot_compile_pool}")
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
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
  endif()
endfunction()

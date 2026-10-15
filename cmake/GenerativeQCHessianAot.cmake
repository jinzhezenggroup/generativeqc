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
  add_library(${target} STATIC ${_sources} "${_registry}" "${PROJECT_SOURCE_DIR}/src/dft/hessian_aot.cpp")
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PUBLIC CUDA::cudart)
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  if(GENERATIVEQC_BUILD_TESTS)
    add_executable(generativeqc_hessian_aot_probe
      "${PROJECT_SOURCE_DIR}/tests/native/hessian_aot_probe.cpp")
    target_link_libraries(generativeqc_hessian_aot_probe PRIVATE ${target})
  endif()
endfunction()

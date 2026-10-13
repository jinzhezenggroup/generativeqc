include_guard(GLOBAL)

# The compiler's stationary MethodIR catalog is the build inventory authority.
# A named profile remains a performance specialization, not method admission.
# "auto" discovers the qualified profiles; "" disables the optional artifacts.
set(GENERATIVEQC_STATIONARY_AOT_PROFILES "auto" CACHE STRING
    "Stationary CUDA AOT specializations: auto, an explicit semicolon list, or empty")

function(generativeqc_select_stationary_aot_profiles output)
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Stationary CUDA profile discovery requires build-time Python")
  endif()
  set(_generativeqc_stationary_discovery_inputs
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/generativeqc_stationary_profile_inputs.txt")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_native_stationary_cuda.py"
            --list-aot-profiles
            --cmake-dependencies "${_generativeqc_stationary_discovery_inputs}"
    OUTPUT_VARIABLE _generativeqc_stationary_profile_catalog
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  # Discovery determines target/output membership, so ordinary content changes
  # to its compiler inputs must regenerate CMake, not just generated CUDA.
  file(STRINGS "${_generativeqc_stationary_discovery_inputs}"
       _generativeqc_stationary_configure_dependencies)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${_generativeqc_stationary_configure_dependencies})
  if(GENERATIVEQC_STATIONARY_AOT_PROFILES STREQUAL "auto")
    set(selected ${_generativeqc_stationary_profile_catalog})
  else()
    set(selected ${GENERATIVEQC_STATIONARY_AOT_PROFILES})
  endif()
  foreach(profile IN LISTS selected)
    if(NOT profile IN_LIST _generativeqc_stationary_profile_catalog)
      message(FATAL_ERROR "Unknown compiler-qualified stationary CUDA AOT profile: ${profile}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES selected)
  list(SORT selected)
  set(${output} "${selected}" PARENT_SCOPE)
endfunction()

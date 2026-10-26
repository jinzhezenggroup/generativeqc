include_guard(GLOBAL)

# The compiler's stationary MethodIR catalog is the build inventory authority.
# A named profile remains a performance specialization, not method admission.
# "auto" follows the curated inventory and selected automatic energy package;
# explicit profiles bound force contents further; "" disables the artifacts.
set(GENERATIVEQC_STATIONARY_AOT_PROFILES "auto" CACHE STRING
    "Stationary CUDA AOT specializations: auto, an explicit semicolon list, or empty")

function(generativeqc_select_stationary_aot_profiles output)
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Stationary CUDA profile discovery requires build-time Python")
  endif()
  set(_generativeqc_stationary_discovery_inputs
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/generativeqc_stationary_profile_inputs.txt")
  set(_generativeqc_stationary_discovery_catalog
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/generativeqc_stationary_profiles.cmake")
  set(_automatic_arguments)
  foreach(functional IN LISTS GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS)
    list(APPEND _automatic_arguments --functional "${functional}")
  endforeach()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_native_stationary_cuda.py"
            --list-available-aot-profiles
            --cmake-dependencies "${_generativeqc_stationary_discovery_inputs}"
            --cmake-catalog "${_generativeqc_stationary_discovery_catalog}"
            ${_automatic_arguments}
    OUTPUT_VARIABLE _generativeqc_stationary_profile_catalog
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  # Discovery determines target/output membership, so ordinary content changes
  # to its compiler inputs must regenerate CMake, not just generated CUDA.
  file(STRINGS "${_generativeqc_stationary_discovery_inputs}"
       _generativeqc_stationary_configure_dependencies)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${_generativeqc_stationary_configure_dependencies})
  include("${_generativeqc_stationary_discovery_catalog}")
  if(NOT GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT)
    set(_generativeqc_automatic_force_catalog)
  endif()
  if(GENERATIVEQC_STATIONARY_AOT_PROFILES STREQUAL "auto")
    set(selected ${_generativeqc_stationary_default_catalog} ${_generativeqc_automatic_force_catalog})
  else()
    set(selected ${GENERATIVEQC_STATIONARY_AOT_PROFILES})
  endif()
  foreach(profile IN LISTS selected)
    if(NOT profile IN_LIST _generativeqc_stationary_profile_catalog)
      message(FATAL_ERROR "Unknown compiler-qualified stationary CUDA AOT profile: ${profile}")
    endif()
    if(profile MATCHES "^libxc_" AND NOT profile IN_LIST _generativeqc_automatic_force_catalog)
      message(FATAL_ERROR "Automatic stationary profile ${profile} requires its matching automatic CUDA XC AOT provider")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES selected)
  list(SORT selected)
  set(${output} "${selected}" PARENT_SCOPE)
endfunction()

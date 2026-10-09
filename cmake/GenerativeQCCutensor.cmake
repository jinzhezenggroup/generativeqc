include_guard(GLOBAL)

# Explicit optional dependency: ordinary CUDA/CPU builds never probe or link
# cuTENSOR. Enabling it supplies capability, not scientific/provider admission.
function(generativeqc_configure_cutensor target)
  if(NOT GENERATIVEQC_ENABLE_CUTENSOR)
    return()
  endif()
  if(NOT GENERATIVEQC_ENABLE_CUDA OR NOT GENERATIVEQC_CUDA_PROVIDER STREQUAL "nvidia")
    message(FATAL_ERROR "cuTENSOR requires the NVIDIA CUDA backend")
  endif()
  if(GENERATIVEQC_PYTHON_WHEEL)
    message(FATAL_ERROR
            "cuTENSOR wheel dependency packaging is not implemented; "
            "use a native build")
  endif()

  set(_generativeqc_cutensor_root "${GENERATIVEQC_CUTENSOR_ROOT}")
  if(NOT _generativeqc_cutensor_root)
    set(_generativeqc_cutensor_root "$ENV{GENERATIVEQC_CUTENSOR_ROOT}")
  endif()
  set(_generativeqc_cutensor_search_options)
  if(_generativeqc_cutensor_root)
    # An explicitly selected SDK must provide *both* headers and library. Never
    # fall through to an unrelated system installation or the prior CMake cache.
    list(APPEND _generativeqc_cutensor_search_options
         HINTS "${_generativeqc_cutensor_root}" NO_DEFAULT_PATH)
  endif()
  # NO_CACHE (CMake >=3.21) makes discovery sensitive to ROOT changes and SDKs
  # replaced in place, rather than accepting stale find_path/find_library cache.
  find_path(_generativeqc_cutensor_include_dir cutensor.h
            ${_generativeqc_cutensor_search_options}
            PATH_SUFFIXES include NO_CACHE REQUIRED)
  find_library(_generativeqc_cutensor_library NAMES cutensor libcutensor.so.2
               ${_generativeqc_cutensor_search_options}
               PATH_SUFFIXES lib lib64 lib/12 lib/12.0 NO_CACHE REQUIRED)
  # find_path() may return a trailing directory slash on some CMake versions;
  # normalize the target property and informational cache value consistently.
  get_filename_component(_generativeqc_cutensor_include_dir
                         "${_generativeqc_cutensor_include_dir}" ABSOLUTE)

  # Keep the historically exposed cache values informational, not authoritative.
  # They must agree with the SDK actually propagated to targets after a ROOT swap.
  set(GENERATIVEQC_CUTENSOR_INCLUDE_DIR "${_generativeqc_cutensor_include_dir}"
      CACHE PATH "Selected cuTENSOR header directory" FORCE)
  set(GENERATIVEQC_CUTENSOR_LIBRARY "${_generativeqc_cutensor_library}"
      CACHE FILEPATH "Selected cuTENSOR shared library" FORCE)

  set(_generativeqc_cutensor_header
      "${_generativeqc_cutensor_include_dir}/cutensor.h")
  foreach(_part MAJOR MINOR)
    file(STRINGS "${_generativeqc_cutensor_header}" _version_lines
         REGEX "^[ \t]*#[ \t]*define[ \t]+CUTENSOR_${_part}[ \t]+[0-9]+([ \t]|$)")
    list(LENGTH _version_lines _version_count)
    if(NOT _version_count EQUAL 1)
      message(FATAL_ERROR
              "Cannot verify cuTENSOR ${_part} version from "
              "${_generativeqc_cutensor_header}")
    endif()
    list(GET _version_lines 0 _version_line)
    if(NOT _version_line MATCHES
       "^[ \t]*#[ \t]*define[ \t]+CUTENSOR_${_part}[ \t]+([0-9]+)([ \t]|$)")
      message(FATAL_ERROR
              "Cannot verify cuTENSOR ${_part} version from "
              "${_generativeqc_cutensor_header}")
    endif()
    set(_cutensor_${_part} "${CMAKE_MATCH_1}")
  endforeach()
  if(NOT _cutensor_MAJOR STREQUAL "2" OR _cutensor_MINOR LESS 8)
    message(FATAL_ERROR
            "The optional native provider requires cuTENSOR 2.8 or later in 2.x")
  endif()
  # A normal incremental build must rerun admission after an in-place upgrade.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${_generativeqc_cutensor_header}" "${_generativeqc_cutensor_library}")

  if(NOT TARGET generativeqc_cutensor)
    add_library(generativeqc_cutensor INTERFACE)
    # An imported target makes CMake pass versioned shared libraries to the
    # linker, instead of asking nvcc to compile an unrecognized .so.2 input.
    add_library(generativeqc_cutensor_library SHARED IMPORTED)
    target_link_libraries(generativeqc_cutensor INTERFACE generativeqc_cutensor_library)
  endif()
  set_target_properties(generativeqc_cutensor PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_generativeqc_cutensor_include_dir}"
    INTERFACE_COMPILE_DEFINITIONS GENERATIVEQC_HAS_CUTENSOR=1)
  set_target_properties(generativeqc_cutensor_library PROPERTIES
                        IMPORTED_LOCATION "${_generativeqc_cutensor_library}")
  # Internal native tests include the same executor types as the library. Keep
  # their macro, headers and link dependency identical to prevent ODR mismatch.
  target_link_libraries(${target} PUBLIC generativeqc_cutensor)
endfunction()

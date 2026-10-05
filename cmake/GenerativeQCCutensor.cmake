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
    message(FATAL_ERROR "cuTENSOR wheel dependency packaging is not implemented; use a native build")
  endif()
  find_path(GENERATIVEQC_CUTENSOR_INCLUDE_DIR cutensor.h
            HINTS "${GENERATIVEQC_CUTENSOR_ROOT}" ENV GENERATIVEQC_CUTENSOR_ROOT
            PATH_SUFFIXES include REQUIRED)
  file(STRINGS "${GENERATIVEQC_CUTENSOR_INCLUDE_DIR}/cutensor.h" _cutensor_major
       REGEX "^#define CUTENSOR_MAJOR +[0-9]+")
  file(STRINGS "${GENERATIVEQC_CUTENSOR_INCLUDE_DIR}/cutensor.h" _cutensor_minor
       REGEX "^#define CUTENSOR_MINOR +[0-9]+")
  if(NOT _cutensor_major MATCHES "CUTENSOR_MAJOR +2([^0-9]|$)" OR
     NOT _cutensor_minor MATCHES "CUTENSOR_MINOR +([0-9]+)")
    message(FATAL_ERROR "The optional native provider requires cuTENSOR 2.8 or later in 2.x")
  endif()
  if(CMAKE_MATCH_1 LESS 8)
    message(FATAL_ERROR "The optional native provider requires cuTENSOR 2.8 or later in 2.x")
  endif()
  # NVIDIA's Python distribution may contain only the versioned SONAME.
  find_library(GENERATIVEQC_CUTENSOR_LIBRARY NAMES cutensor libcutensor.so.2
               HINTS "${GENERATIVEQC_CUTENSOR_ROOT}" ENV GENERATIVEQC_CUTENSOR_ROOT
               PATH_SUFFIXES lib lib64 lib/12 lib/12.0 REQUIRED)
  if(NOT TARGET generativeqc_cutensor)
    add_library(generativeqc_cutensor INTERFACE)
    target_include_directories(generativeqc_cutensor INTERFACE "${GENERATIVEQC_CUTENSOR_INCLUDE_DIR}")
    target_compile_definitions(generativeqc_cutensor INTERFACE GENERATIVEQC_HAS_CUTENSOR=1)
    # An imported target makes CMake pass versioned shared libraries to the
    # linker, instead of asking nvcc to compile an unrecognized .so.2 input.
    add_library(generativeqc_cutensor_library SHARED IMPORTED)
    set_target_properties(generativeqc_cutensor_library PROPERTIES
                          IMPORTED_LOCATION "${GENERATIVEQC_CUTENSOR_LIBRARY}")
    target_link_libraries(generativeqc_cutensor INTERFACE generativeqc_cutensor_library)
  endif()
  # Internal native tests include the same executor types as the library. Keep
  # their macro, headers and link dependency identical to prevent ODR mismatch.
  target_link_libraries(${target} PUBLIC generativeqc_cutensor)
endfunction()
